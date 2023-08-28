#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <chrono>
#include <exception>
#include <cuda.h>
#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <fstream>
#include <string>
#include <cub/cub.cuh>
#include <unordered_map>
// class TEMPException : public std::exception
// {
// public:
//     TEMPException(const char* fl, const char* fn, int ln, int st, const char* msg, const char* nm)
//         : file(fl)
//         , function(fn)
//         , line(ln)
//         , status(st)
//         , message(msg)
//         , name(nm)
//     {
//     }
//     virtual void log(std::ostream& logStream) const;
//     void setMessage(const char* msg) { message = msg; }

// protected:
//     const char* file{nullptr};
//     const char* function{nullptr};
//     int line{0};
//     int status{0};
//     const char* message{nullptr};
//     const char* name{nullptr};
// };

// class CudaError : public TEMPException
// {
// public:
//     CudaError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
//         : TEMPException(fl, fn, ln, stat, msg, "Cuda")
//     {
//     }
// };


// class CublasError : public TEMPException
// {
// public:
//     CublasError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
//         : TEMPException(fl, fn, ln, stat, msg, "cuBLAS")
//     {
//     }
// };
// // break-pointable
// void throwCudaError(const char* file, const char* function, int line, int status, const char* msg)
// {
//     CudaError error(file, function, line, status, msg);
//     throw error;
// }

// // break-pointable
// void throwCublasError(const char* file, const char* function, int line, int status, const char* msg)
// {
//     if (msg == nullptr)
//     {
//         auto s_ = static_cast<cublasStatus_t>(status);
//         switch (s_)
//         {
//         case CUBLAS_STATUS_SUCCESS: msg = "CUBLAS_STATUS_SUCCESS"; break;
//         case CUBLAS_STATUS_NOT_INITIALIZED: msg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
//         case CUBLAS_STATUS_ALLOC_FAILED: msg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
//         case CUBLAS_STATUS_INVALID_VALUE: msg = "CUBLAS_STATUS_INVALID_VALUE"; break;
//         case CUBLAS_STATUS_ARCH_MISMATCH: msg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
//         case CUBLAS_STATUS_MAPPING_ERROR: msg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
//         case CUBLAS_STATUS_EXECUTION_FAILED: msg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
//         case CUBLAS_STATUS_INTERNAL_ERROR: msg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
//         case CUBLAS_STATUS_NOT_SUPPORTED: msg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
//         case CUBLAS_STATUS_LICENSE_ERROR: msg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
//         }
//     }
//     CublasError error(file, function, line, status, msg);
//     throw error;
// }

// #ifdef _MSC_VER
// #define FN_NAME __FUNCTION__
// #else
// #define FN_NAME __func__
// #endif
// #define PLUGIN_CUASSERT(status_)                                                                                   \
// {                                                                                                                  \
//     auto s_ = status_;                                                                                             \
//     if (s_ != cudaSuccess)                                                                                         \
//     {                                                                                                              \
//         const char* msg = cudaGetErrorString(s_);                                                                  \
//         throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                                       \
//     }                                                                                                              \
// }

// #define PLUGIN_CUBLASASSERT(status_)                                                                               \
// {                                                                                                                  \
//     auto s_ = status_;                                                                                             \
//     if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
//     {                                                                                                              \
//         throwCublasError(__FILE__, FN_NAME, __LINE__, s_, nullptr);                                                         \
//     }                                                                                                              \
// }
#define PLUGIN_CUBLASASSERT(status_) status_
#define PLUGIN_CUASSERT(status_) status_
using half = __half;

constexpr size_t maxWorkspaceBytes = 4194304; // 4MB
size_t getWorkspaceSize()
{
    return maxWorkspaceBytes;
}

/* Structure to store information about different run trials */
typedef struct customMatMultPerfType_t
{
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time{1000000.F};
    size_t workspaceSize; // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

static inline bool time_compare(const customMatmulPerf_t& perf_a, const customMatmulPerf_t& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}
/* CAUTION : must match cublasLtMatmulTile_t */
const char* const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8",
    "8x32",
    "16x16",
    "32x8",
    "8x64",
    "16x32",
    "32x16",
    "64x8",
    "32x32",
    "32x64",
    "64x32",
    "32x128",
    "64x64",
    "128x32",
    "64x128",
    "128x64",
    "64x256",
    "128x128",
    "256x64",
    "64x512",
    "128x256",
    "256x128",
    "512x64",
};

struct AlgoProps
{
    int algoId;
    int tile;
    int swizzle;
    int customOption;
    int numSplitsK;
    int reductionScheme;
    int mathMode;

    void populate(const cublasLtMatmulAlgo_t& algo)
    {
        const cublasLtMatmulAlgo_t* matmulAlgo = &algo;
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CAP_MATHMODE_IMPL, &mathMode, sizeof(mathMode), nullptr));
    }
};

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(const customMatmulPerf_t& perf, int const& m, int const& n, int const& k)
{
    AlgoProps p;
    p.populate(perf.algo);
    /* Calculate GFLOPS */
    double timeAvg = perf.time * 1e-3; // Convert to seconds. It has been divided by kernelRepeats in customMatmulRun().
    double gflop = (2 * static_cast<unsigned long long int>(m * n) * k) * 1e-9; // Real

    std::cout << "Algo=" << p.algoId << " Tile=" << p.tile << " (" << matmulTileName[p.tile] << ") K=" << p.numSplitsK << " Red.Sch.=" << p.reductionScheme << " Swiz=" << p.swizzle << " Cust=" << p.customOption << " Stat=" << perf.status << " Time=" << perf.time << " WSbytes=" << perf.workspaceSize << " math=" << p.mathMode << " waves=" << perf.wavesCount << " GFlops=" << (gflop / timeAvg) << std::endl;
}

template <typename T>
struct GemmTypes
{
};

template <>
struct GemmTypes<half>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_16F;
    using dataTypeI = half;
    static const cudaDataType_t cudaTypeO = CUDA_R_16F;
    using dataTypeO = half;
    static const cudaDataType_t cudaTypeS = CUDA_R_16F;
    using dataTypeS = half;
#if CUBLAS_VER_MAJOR < 11
    static const cudaDataType_t cudaTypeCom = CUDA_R_16F;
#else
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_16F;
#endif
};

template <>
struct GemmTypes<float>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_32F;
    using dataTypeI = float;
    static const cudaDataType_t cudaTypeO = CUDA_R_32F;
    using dataTypeO = float;
    static const cudaDataType_t cudaTypeS = CUDA_R_32F;
    using dataTypeS = float;
#if CUBLAS_VER_MAJOR < 11
    static const cudaDataType_t cudaTypeCom = CUDA_R_32F;
#else
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_32F;
#endif
};

template <typename T>
struct Gemm
{
    using Types = GemmTypes<T>;
    typename Types::dataTypeI* A{nullptr};
    typename Types::dataTypeI* B{nullptr};
    typename Types::dataTypeO* C{nullptr};
    int m, n, k, ldA, ldB, ldC, rA, rB, rC, cA, cB, cC;
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;

    size_t elemA;
    size_t elemB;
    size_t elemC;
    bool transA, transB;

    cublasOperation_t opA;
    cublasOperation_t opB;

    const int word_size{sizeof(T)};
    typename Types::dataTypeS alpha;
    typename Types::dataTypeS beta;

    Gemm() {}

    Gemm(int m_, int n_, int k_, bool tA, bool tB)
    {
        init(m_, n_, k_, tA, tB);
    }

    void init(int m_, int n_, int k_, bool tA, bool tB) noexcept
    {
        m = m_;
        n = n_;
        k = k_;
        transA = tA;
        transB = tB;
        ldA = transA ? k : m;
        ldB = transB ? n : k;
        ldC = m;

        rA = ldA;
        rB = ldB;
        rC = ldC;

        cA = transA ? m : k;
        cB = transB ? k : n;
        cC = n;

        opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        elemA = m * k;
        elemB = n * k;
        elemC = n * m;
        bytesA = word_size * elemA;
        bytesB = word_size * elemB;
        bytesC = word_size * elemC;
        alpha = T(1.f);
        beta = T(0.f);
    }
};

auto constexpr algoCombinations = 6000;
auto constexpr algoIds = 40;
auto constexpr printAlgos = 1;
auto constexpr kernelRepeats = 10;
auto constexpr threadsPerBlock = 1024;
static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle, // to get the capabilities (required a GPU)
    cublasLtMatmulDesc_t operationDesc, void const* alpha,       /* host or device pointer */
    void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc,
    void const* beta, /* host or device pointer */
    void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulAlgo_t const& algo, void* workSpace, size_t workSpaceSizeInBytes, customMatmulPerf_t& perfResults,
    cudaStream_t stream, cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{

    cublasLtMatmulHeuristicResult_t heurResult;

    /* Looping over the Algo */
    cublasStatus_t algoStatus
        = cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS)
    {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes)
        {
            if (cudaEventRecord(startEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            for (int loop = 0; loop < kernelRepeats; loop++)
            {
                cublasStatus_t oneRunStatus
                    = cublasLtMatmul(ltHandle, operationDesc, alpha, /* host or device pointer */
                        A, Adesc, B, Bdesc, beta,                    /* host or device pointer */
                        C, Cdesc, D, Ddesc, &algo, workSpace, workSpaceSizeInBytes, stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS)
                {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            if (cudaEventRecord(stopEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            if (cudaEventSynchronize(stopEvent) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            float time;
            if (cudaEventElapsedTime(&time, startEvent, stopEvent) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            perfResults.algo = algo;
            perfResults.time = time / kernelRepeats; // Average time
            perfResults.workspaceSize = heurResult.workspaceSize;
            perfResults.wavesCount = heurResult.wavesCount;
        }
        else
        {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; // Not enough workspace
        }
    }
    return algoStatus;
}


// Sample wrapper running through multiple algo and config attributes
// combination for single precision gemm using cublasLt low-level API
void LtGemmSearch(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int const& m,
    int const& n, int const& k, void const* alpha,                                  /* host pointer */
    void const* A, int const& lda, void const* B, int const& ldb, void const* beta, /* host pointer */
    void* C, int const& ldc, void* workSpace, size_t workSpaceSize,
#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t computeType,
#else
    cublasComputeType_t computeType,
#endif
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
    std::vector<customMatmulPerf_t>& perfResults)
{

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
    cudaStream_t stream = nullptr;

    // SplitK value that we are going to try when SplitK is supported for a given
    // algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

    // Let try a fixed number of combinations
    int algoCount = 0;
    int nbAlgoIds = 0;
    int algoIdA[algoIds];
    // customMatmulPerf_t perfResults[algoCombinations];

    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceCreate(&preference));
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize)));

    const int mathMode = Ctype == CUDA_R_16F ? 1 : 0;
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MATH_MODE_MASK, &mathMode, sizeof(mathMode)));
    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details
    // about defaults; here we just need to set the transforms for A and B
#if CUBLAS_VER_MAJOR < 11
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, computeType));
#else
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#endif
    PLUGIN_CUBLASASSERT(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    PLUGIN_CUBLASASSERT(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // Create matrix descriptors. We are good with the details here so no need to
    // set any extra attributes
    PLUGIN_CUBLASASSERT(
        cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    PLUGIN_CUBLASASSERT(
        cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType =
    // Atype = Btype = Ctype = Dtype = CUDA_R_32F)
    PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIds, algoIdA, &nbAlgoIds));


    // Create CUDA event to time the execution time of each algo
    PLUGIN_CUASSERT(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    PLUGIN_CUASSERT(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (algoCount < algoCombinations); idx++)
    {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status
            = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            continue;
        }

        int mathMode = -1;
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_MATHMODE_IMPL, &mathMode, sizeof(mathMode), nullptr));
        // TODO is this the right way to check that it's SGEMM?
        if (Ctype == CUDA_R_32F && mathMode == 1)
        {
            // if mathMode is 1, cublasLt chooses automatically to run in mixed precision for certain sizes
            continue;
        }

        // Query the tiles enums supported by that algo
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
        int nbTiles = int(sizeWritten / sizeof(int));
        int* tileA = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0)
        {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        int splitkSupport, redMask, swizzlingMax, customOptionMax, epilogueMask;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the
        // different combinations
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten));

        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten));

        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++)
        {
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++)
            {
                PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
                /* Loop over the CTAs swizzling support */
                for (int k = 0; k <= swizzlingMax; k++)
                {
                    int splitK_trial = 0;
                    if (splitkSupport)
                    {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in
                    // addition to the case where splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (algoCount < algoCombinations); l++)
                    {
                        /* Setup attribute of the algo to run */
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx])));
                        int splitK_val = 0;
                        int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int)));

                        if (l > 0)
                        { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            PLUGIN_CUBLASASSERT(
                                cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                    &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1])));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1; redScheme < static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK)
                                 && (algoCount < algoCombinations);
                                 redScheme = redScheme << 1)
                            {
                                if (redScheme & redMask)
                                {
                                    PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme)));

                                    status
                                        = customMatmulRun(ltHandle, operationDesc, alpha, /* host or device pointer */
                                            A, Adesc, B, Bdesc, beta,                     /* host or device pointer */
                                            C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount],
                                            stream, startEvent, stopEvent);
                                    perfResults[algoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS)
                                    {

                                        algoCount++;
                                    }
                                } // end if
                            }     // end for
                        }
                        else
                        { // Non-splitK case
                            /* if user preference is ok with workspace */
                            if (algoCount < algoCombinations)
                            {
                                status = customMatmulRun(ltHandle, operationDesc, alpha, /* host or device pointer */
                                    A, Adesc, B, Bdesc, beta,                            /* host or device pointer */
                                    C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount], stream,
                                    startEvent, stopEvent);
                                perfResults[algoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS)
                                    algoCount++;
                            }
                        }
                    } // end l
                }     // end k
            }         // end customOption
        }             // end tileIdx
        delete[] tileA;
    } // end idx

    // Sort the results per run duration
    std::sort(perfResults.begin(), perfResults.end(), time_compare);
    // Print timing and perf details of the fastest combinations
    // for (int i = 0; i < perfResults.size(); i++){
    for (int i = 0; i < printAlgos; i++)
    {
        if (perfResults[i].time == 1000000.F)
            break;
        printPerfStructure(perfResults[i], m, n, k);
    }

    // Descriptors are no longer needed as all GPU work was already enqueued
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceDestroy(preference));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Cdesc));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Bdesc));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Adesc));
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescDestroy(operationDesc));
    PLUGIN_CUASSERT(cudaEventDestroy(startEvent));
    PLUGIN_CUASSERT(cudaEventDestroy(stopEvent));
}
template <typename T>
void LtGemmSearch(cublasLtHandle_t ltHandle, const Gemm<T>& g, void* workSpace, size_t workSpaceSize,
    std::vector<customMatmulPerf_t>& perfResults)
{
    // clang-format off
  LtGemmSearch(ltHandle,
               g.opA,
               g.opB,
               g.m,
               g.n,
               g.k,
               &g.alpha,
               g.A,
               g.ldA,
               g.B,
               g.ldB,
               &g.beta,
               g.C,
               g.ldC,
               workSpace,
               workSpaceSize,
               Gemm<T>::Types::cudaTypeCom,
               Gemm<T>::Types::cudaTypeS,
               Gemm<T>::Types::cudaTypeI,
               Gemm<T>::Types::cudaTypeI,
               Gemm<T>::Types::cudaTypeO,
               perfResults);
    // clang-format on
}

struct LtContext
{
    cublasLtHandle_t cublas{nullptr};
    cudaDataType_t typeA;
    cudaDataType_t typeB;
    cudaDataType_t typeC;
#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t typeComp;
#else
    cublasComputeType_t typeComp;
#endif
    cudaDataType_t typeS;
    cublasLtMatmulDesc_t operationDesc{nullptr};
    cublasLtMatrixLayout_t Adesc{nullptr};
    cublasLtMatrixLayout_t Bdesc{nullptr};
    cublasLtMatrixLayout_t Cdesc{nullptr};
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    void attach()
    {
        PLUGIN_CUBLASASSERT(cublasLtCreate(&cublas));
    }

    void detach()
    {
        PLUGIN_CUBLASASSERT(cublasLtDestroy(cublas));
    }

    void destroy()
    {
        if (operationDesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatmulDescDestroy(operationDesc));
            operationDesc = nullptr;
        }
        if (Adesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Adesc));
            Adesc = nullptr;
        }
        if (Bdesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Bdesc));
            Bdesc = nullptr;
        }
        if (Cdesc)
        {
            PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Cdesc));
            Cdesc = nullptr;
        }
    }

    template <typename T>
    void create(Gemm<T>& g, size_t workspaceSize)
    {
        typeA = Gemm<T>::Types::cudaTypeI;
        typeB = Gemm<T>::Types::cudaTypeI;
        typeC = Gemm<T>::Types::cudaTypeO;
        typeS = Gemm<T>::Types::cudaTypeS;
        typeComp = Gemm<T>::Types::cudaTypeCom; // compute

        // OPERATION
#if CUBLAS_VER_MAJOR < 11
        PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, typeComp));
#else
        PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, typeComp, typeS));
#endif
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &g.opA, sizeof(g.opA)));
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &g.opB, sizeof(g.opB)));

        // MAT DESC
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Adesc, typeA, g.rA, g.cA, g.ldA));
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Bdesc, typeB, g.rB, g.cB, g.ldB));
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Cdesc, typeC, g.rC, g.cC, g.ldC));
    }

    void setN(uint64_t n)
    {
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
        PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
    }
};

template <typename T>
cublasStatus_t inline cublasLtMatmul(
    LtContext& ctx, Gemm<T>& g, cublasLtMatmulAlgo_t algo, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // clang-format off
     return cublasLtMatmul(ctx.cublas,
                    ctx.operationDesc,
                    &g.alpha,
                    g.A,
                    ctx.Adesc,
                    g.B,
                    ctx.Bdesc,
                    &g.beta,
                    g.C,
                    ctx.Cdesc,
                    g.C,
                    ctx.Cdesc,
                    &algo,
                    workspace,
                    workspaceSize,
                    stream
                    );
    // clang-format on
}

template <typename T>
inline cublasLtMatmulAlgo_t gemmSearch(
    const int m, const int n, const int k, const size_t workspaceSize, size_t& actualWorkspace)
{

    Gemm<T> g(m, n, k, false, false);
    std::vector<customMatmulPerf_t> perfResults(algoCombinations);

    PLUGIN_CUASSERT(cudaMalloc(reinterpret_cast<void**>(&g.A), g.bytesA));
    PLUGIN_CUASSERT(cudaMalloc(reinterpret_cast<void**>(&g.B), g.bytesB));
    PLUGIN_CUASSERT(cudaMalloc(reinterpret_cast<void**>(&g.C), g.bytesC));

    void* workspace;
    PLUGIN_CUASSERT(cudaMalloc(&workspace, workspaceSize));
    cublasLtHandle_t lt;
    PLUGIN_CUBLASASSERT(cublasLtCreate(&lt));
    LtGemmSearch(lt, g, workspace, workspaceSize, perfResults);
    PLUGIN_CUASSERT(cudaDeviceSynchronize());
    PLUGIN_CUBLASASSERT(cublasLtDestroy(lt));
    PLUGIN_CUASSERT(cudaFree(workspace));

    PLUGIN_CUASSERT(cudaFree(g.A));
    PLUGIN_CUASSERT(cudaFree(g.B));
    PLUGIN_CUASSERT(cudaFree(g.C));

    actualWorkspace = perfResults[0].workspaceSize;
    return perfResults[0].algo;
}

template <typename T>
inline cublasLtMatmulAlgo_t gemmSearch(Gemm<T>& g, const size_t workspaceSize, size_t& actualWorkspace)
{

    std::vector<customMatmulPerf_t> perfResults(algoCombinations);

    PLUGIN_CUASSERT(cudaMalloc(&g.A, g.bytesA));
    PLUGIN_CUASSERT(cudaMalloc(&g.B, g.bytesB));
    PLUGIN_CUASSERT(cudaMalloc(&g.C, g.bytesC));

    void* workspace;
    PLUGIN_CUASSERT(cudaMalloc(&workspace, workspaceSize));
    cublasLtHandle_t lt;
    PLUGIN_CUBLASASSERT(cublasLtCreate(&lt));
    LtGemmSearch(lt, g, workspace, workspaceSize, perfResults);
    PLUGIN_CUASSERT(cudaDeviceSynchronize());
    PLUGIN_CUBLASASSERT(cublasLtDestroy(lt));
    PLUGIN_CUASSERT(cudaFree(workspace));

    PLUGIN_CUASSERT(cudaFree(g.A));
    PLUGIN_CUASSERT(cudaFree(g.B));
    PLUGIN_CUASSERT(cudaFree(g.C));

    actualWorkspace = perfResults[0].workspaceSize;
    return perfResults[0].algo;
}

using namespace std::chrono;

double time_sec(high_resolution_clock::time_point st, high_resolution_clock::time_point ed){
    duration<double,std::ratio<1,1>> duration_s(ed-st);
    return duration_s.count();
}
double time_milisec(high_resolution_clock::time_point st, high_resolution_clock::time_point ed){
    duration<double,std::ratio<1,1000>> duration_ms = duration_cast<duration<double,std::ratio<1,1000>>>(ed-st);
    return duration_ms.count();
}

template <typename T>
class FCGemm{};

template <>
class FCGemm<half>{
public:
    FCGemm(){

    }
    void setup(at::Tensor x, at::Tensor W){
        BS = x.size(0);
        L = x.size(1);
        M = BS * L;
        K = x.size(2);
        N = W.size(1);
        torch::TensorOptions ws_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        const size_t workspaceSize = getWorkspaceSize();
        workSpace = static_cast<float *>(torch::zeros({maxWorkspaceBytes/4}, ws_options).data_ptr<float>());
        Gemm<half> g(M, N, K, false, false);
        mLtContext.attach();
        mLtContext.destroy();
        mLtContext.create(g, maxWorkspaceBytes);
        size_t actualWorkspace = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        mAlgo = gemmSearch<half>(M, N, K, maxWorkspaceBytes, actualWorkspace);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("Searching best gemm Algo, time= %fms\n", time_milisec(t0, t1));
    }
    int forward(half const* input, half const* weight, half* output, int32_t M, int32_t N, int32_t K){
        Gemm<half> g(M, N, K, false, false);
        mLtContext.setN(static_cast<uint64_t>(N));
        g.A = const_cast<half*>(input);
        g.B = const_cast<half*>(weight);
        g.C = output;
        int status = cublasLtMatmul(mLtContext, g, mAlgo, workSpace, workspaceSize, 0);
        return status;
    }
    float* workSpace;
    size_t workspaceSize;
    // Gemm<T> g;
    LtContext mLtContext;
    int32_t BS, L, M, K, N;
    cublasLtMatmulAlgo_t mAlgo;
};
std::unordered_map<std::string, FCGemm<half>*> gemmCache;
cudaEvent_t   start, stop;
at::Tensor ffc_forward_cuda(
    at::Tensor x,
    at::Tensor W,
    std::string &fcid
){
    int32_t const BS = x.size(0);
    int32_t const L = x.size(1);
    int32_t const M = BS * L;
    int32_t const K = x.size(2);
    int32_t const N = W.size(1);
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;

    auto output_tensor = at::zeros({BS, L, N}, x.options());
    if (x.dtype() == at::kHalf)
    {
        auto gemmfunc_it = gemmCache.find(fcid);
        if(gemmfunc_it == gemmCache.end()){
            FCGemm<half> *p_gemmfunc__ = new FCGemm<half>();
            p_gemmfunc__->setup(x, W);
            gemmCache[fcid] = p_gemmfunc__;
            gemmfunc_it = gemmCache.find(fcid);
        }
        FCGemm<half> *p_gemmfunc = gemmfunc_it->second;
        auto t1 = std::chrono::high_resolution_clock::now();
        auto const input = reinterpret_cast<half *>(x.data_ptr<at::Half>());
        auto const weight = reinterpret_cast<half *>(W.data_ptr<at::Half>());
        auto output = reinterpret_cast<half*>(output_tensor.data_ptr<at::Half>());
        auto t2 = std::chrono::high_resolution_clock::now();

        // cudaEventRecord(start, 0);
        p_gemmfunc->forward(input, weight, output, M, N, K);
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // float   elapsedTime;
        // cudaEventElapsedTime( &elapsedTime,start, stop);
        printf("set up: %fms, cublasMatmul: %.4fms\n", 
                time_milisec(t1,t2), 0.);
    }
    else
    {
        TORCH_CHECK(false, "float not supported!");
    }
    return output_tensor;
}
// at::Tensor ffc_forward_cuda(
//     at::Tensor x,
//     at::Tensor W,
//     std::string &fcid
// ){
//     int32_t const BS = x.size(0);
//     int32_t const L = x.size(1);
//     int32_t const M = BS * L;
//     int32_t const K = x.size(2);
//     int32_t const N = W.size(1);
//     cudaEventCreate( &start );
//     cudaEventCreate( &stop ) ;

//     torch::TensorOptions ws_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
//     const size_t workspaceSize = getWorkspaceSize();
//     auto workSpace = static_cast<float *>(torch::zeros({2*32*32}, ws_options).data_ptr<float>());

//     auto output_tensor = at::empty({BS, L, N}, x.options());
//     cublasLtMatmulAlgo_t mAlgo;
//     if (x.dtype() == at::kHalf)
//     {
//         Gemm<half> g(M, N, K, false, true);
//         auto mAlgo_it = gemmCache.find(fcid);
//         auto t0 = std::chrono::high_resolution_clock::now();
//         if(mAlgo_it == gemmCache.end()){
//             mLtContext.destroy();
//             mLtContext.create(g, maxWorkspaceBytes);
//             size_t actualWorkspace = 0;
//             mAlgo = gemmSearch<half>(M, N, K, maxWorkspaceBytes, actualWorkspace);
//             gemmCache[fcid] = mAlgo;
//             printf("Searching best gemm Algo\n");
//         } else {
//             printf("Found best matching gemm Algo\n");
//             mAlgo = mAlgo_it->second;
//         }
//         auto t1 = std::chrono::high_resolution_clock::now();
//         auto const input = reinterpret_cast<half *>(x.data_ptr<at::Half>());
//         auto const weight = reinterpret_cast<half *>(W.data_ptr<at::Half>());
//         auto output = reinterpret_cast<half*>(output_tensor.data_ptr<at::Half>());

//         mLtContext.setN(static_cast<uint64_t>(M));
//         g.A = const_cast<half*>(input);
//         g.B = const_cast<half*>(weight);
//         g.C = output;
//         auto t2 = std::chrono::high_resolution_clock::now();
//         cudaEventRecord(start, 0) ;
//         int status = cublasLtMatmul(mLtContext, g, mAlgo, workSpace, workspaceSize, 0);
//         cudaEventRecord(stop, 0);
//         cudaEventSynchronize(stop);
//         float   elapsedTime;
//         cudaEventElapsedTime( &elapsedTime,start, stop);
//         auto t3 = std::chrono::high_resolution_clock::now();
//         printf("search: %fms, set up: %fms, cublasMatmul: %.4fms,  %fms(by cpu)\n", 
//                 time_milisec(t0,t1), time_milisec(t1,t2), elapsedTime, time_milisec(t2,t3));
//     }
//     else
//     {
//         mLtContext.destroy();
//         Gemm<float> gg(M, N, K, false, true);
//         mLtContext.create(gg, maxWorkspaceBytes);
//         size_t actualWorkspace = 0;
//         cublasLtMatmulAlgo_t mAlgo = gemmSearch<float>(M,N,K, maxWorkspaceBytes, actualWorkspace);
//         auto const input = reinterpret_cast<float *>(x.data_ptr<float>());
//         auto const weight = reinterpret_cast<float *>(W.data_ptr<float>());
//         auto output = reinterpret_cast<float*>(output_tensor.data_ptr<float>());

//         Gemm<float> g(M, N, K, false, true);
//         g.A = const_cast<float*>(input);
//         g.B = const_cast<float*>(weight);
//         g.C = output;
//         int status = cublasLtMatmul(mLtContext, g, mAlgo, workSpace, workspaceSize, 0);
//     }
//     return output_tensor;
// }