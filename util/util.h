﻿#ifndef __H_util_H__
#define __H_util_H__
#include <stdio.h>
#include "flags.h"
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// PRETTY PRINT COLORS //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#define __RESET       ""                     /* Default*/
#define __BLACK       ""                     /* Black */
#define __RED         ""                     /* Red */
#define __GREEN       ""                     /* Green */
#define __YELLOW      ""                     /* Yellow */
#define __BLUE        ""                     /* Blue */
#define __MAGENTA     ""                     /* Magenta */
#define __CYAN        ""                     /* Cyan */
#define __WHITE       ""                     /* White */
#define __BOLDBLACK   ""                     /* Bold Black */
#define __BOLDRED     ""                     /* Bold Red */
#define __BOLDGREEN   ""                     /* Bold Green */
#define __BOLDYELLOW  ""                     /* Bold Yellow */
#define __BOLDBLUE    ""                     /* Bold Blue */
#define __BOLDMAGENTA ""                     /* Bold Magenta */
#define __BOLDCYAN    ""                     /* Bold Cyan */
#define __BOLDWHITE   ""                     /* Bold White */
#else
// The following are UNIX ONLY terminal color codes.
// ref: http://stackoverflow.com/questions/9158150/colored-output-in-c
#define __RESET       "\033[0m"              /* Default */
#define __BLACK       "\033[30m"             /* Black */
#define __RED         "\033[31m"             /* Red */
#define __GREEN       "\033[32m"             /* Green */
#define __YELLOW      "\033[33m"             /* Yellow */
#define __BLUE        "\033[34m"             /* Blue */
#define __MAGENTA     "\033[35m"             /* Magenta */
#define __CYAN        "\033[36m"             /* Cyan */
#define __WHITE       "\033[37m"             /* White */
#define __BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define __BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define __BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define __BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define __BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define __BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define __BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define __BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////// ALWAYS DEFINED (ASSERT, EXPECT) /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#define ASSERT(statement) \
  do { \
    if (!(statement)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED" << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define ASSERT_EQ(a, b) \
  do { \
    if ((a) != (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define ASSERT_EQ_EPS(a, b, eps)  \
  do { \
    if ((a) - (b) > eps || (b) - (a) > eps) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define ASSERT_GEQ(a, b) \
  do { \
    if ((a) < (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " >= " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define ASSERT_GT(a, b) \
  do { \
    if ((a) <= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " > " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)


#define ASSERT_LEQ(a, b) \
  do { \
    if ((a) > (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " <= " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define ASSERT_LT(a, b) \
  do { \
    if ((a) >= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " < " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define ASSERT_NEQ(a, b) \
  do { \
    if ((a) == (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " != " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#if ALSM_USE_CUDA
#define CUDA_CHECK_ERR(a) \
  do { \
    cudaError_t err = a; \
    if (err != cudaSuccess) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ERROR_CUDA: " << cudaGetErrorString(err) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)
__DEVICE__ const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}
#define CUBLAS_CHECK_ERR(a) \
  do { \
    cublasStatus_t err = a; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ERROR_CUBLAS: " << cublasGetErrorString(err) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#endif

#define EXPECT(statement) \
  do { \
    if (!(statement)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED" << __RESET << std::endl; \
		    } \
      } while (0)

#define EXPECT_EQ(a, b) \
  do { \
    if ((a) != (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define EXPECT_EQ_EPS(a, b, eps)  \
  do { \
    if ((a) - (b) > eps || (b) - (a) > eps) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)


#define EXPECT_GEQ(a, b) \
  do { \
    if ((a) < (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " >= " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define EXPECT_GT(a, b) \
  do { \
    if ((a) <= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " > " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define EXPECT_LEQ(a, b) \
  do { \
    if ((a) > (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " <= " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define EXPECT_LT(a, b) \
  do { \
    if ((a) >= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " < " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define EXPECT_NEQ(a, b) \
  do { \
    if ((a) == (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " != " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)


////////////////////////////////////////////////////////////////////////////////
/////////////////////////// -DDEBUG DEFINED ONLY ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#ifndef DEBUG

#define DEBUG_ASSERT(statement)                  do { } while (0)
#define DEBUG_ASSERT_EQ(a, b)                    do { } while (0)
#define DEBUG_ASSERT_EQ_EPS(a, b, tol)           do { } while (0)
#define DEBUG_ASSERT_GEQ(a, b)                   do { } while (0)
#define DEBUG_ASSERT_GT(a, b)                    do { } while (0)
#define DEBUG_ASSERT_LEQ(a, b)                   do { } while (0)
#define DEBUG_ASSERT_LT(a, b)                    do { } while (0)
#define DEBUG_ASSERT_NEQ(a, b)                   do { } while (0)
#ifdef __CUDACC__
#define DEBUG_CUDA_CHECK_ERR()                   do { } while (0)
#endif
#define DEBUG_EXPECT(statement)                  do { } while (0)
#define DEBUG_EXPECT_EQ(a, b)                    do { } while (0)
#define DEBUG_EXPECT_EQ_EPS(a, b, tol)           do { } while (0)
#define DEBUG_EXPECT_GEQ(a, b)                   do { } while (0)
#define DEBUG_EXPECT_GT(a, b)                    do { } while (0)
#define DEBUG_EXPECT_LEQ(a, b)                   do { } while (0)
#define DEBUG_EXPECT_LT(a, b)                    do { } while (0)
#define DEBUG_EXPECT_NEQ(a, b)                   do { } while (0)
#define DEBUG_PRINTF(message, ...)               do { } while (0)
#define DEBUG_PRINT(message)                     do { } while (0)
#define DEBUG_PRINTF_IF(statement, message, ...) do { } while (0)
#define DEBUG_PRINT_IF(statement, message, ...)  do { } while (0)

#else

#define DEBUG_ASSERT(statement) \
  do { \
    if (!(statement)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED" << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_EQ(a, b) \
  do { \
    if ((a) != (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_EQ_EPS(a, b, eps)  \
  do { \
    if ((a) - (b) > eps || (b) - (a) > eps) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_GEQ(a, b) \
  do { \
    if ((a) < (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " >= " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_GT(a, b) \
  do { \
    if ((a) <= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " > " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_LEQ(a, b) \
  do { \
    if ((a) > (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " <= " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_LT(a, b) \
  do { \
    if ((a) >= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " < " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#define DEBUG_ASSERT_NEQ(a, b) \
  do { \
    if ((a) == (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ASSERT_FAILED: " << (a) << " != " << (b) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)

#if ALSM_USE_CUDA
#define DEBUG_CUDA_CHECK_ERR() \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "ERROR_CUDA: " << cudaGetErrorString(err) \
                << __RESET << std::endl; \
      exit(EXIT_FAILURE); \
		    } \
      } while (0)
#endif

#define DEBUG_EXPECT(statement) \
  do { \
    if (!(statement)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED" << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_EXPECT_EQ(a, b) \
  do { \
    if ((a) != (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_EXPECT_EQ_EPS(a, b, eps)  \
  do { \
    if ((a) - (b) > eps || (b) - (a) > eps) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " == " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)
#define DEBUG_EXPECT_GEQ(a, b) \
  do { \
    if ((a) < (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " >= " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_EXPECT_GT(a, b) \
  do { \
    if ((a) <= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " > " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_EXPECT_LEQ(a, b) \
  do { \
    if ((a) > (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " <= " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_EXPECT_LT(a, b) \
  do { \
    if ((a) >= (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " < " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_EXPECT_NEQ(a, b) \
  do { \
    if ((a) == (b)) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __BLUE << __func__ << "\n" \
                << __RED << "EXPECT_FAILED: " << (a) << " != " << (b) \
                << __RESET << std::endl; \
		    } \
      } while (0)

#define DEBUG_PRINTF(message, ...) \
  do { \
    printf("%s:%d:%s%s\n%sMESSAGE: ", __FILE__,  __LINE__, __BLUE, __func__, \
        __GREEN); \
    printf(message, __VA_ARGS__); \
    printf("%s\n", __RESET); \
      } while (0)

#define DEBUG_PRINT(message) \
  do { \
    printf("%s:%d:%s%s\n%sMESSAGE: %s%s\n", __FILE__,  __LINE__, __BLUE, \
        __func__, __GREEN, message, __RESET); \
      } while (0)

#define DEBUG_PRINTF_IF(statement, message, ...) \
  do { \
    if ((statement)) { \
      printf("%s:%d:%s%s\n%sMESSAGE: ", __FILE__,  __LINE__, __BLUE, __func__, \
          __GREEN); \
      printf(message, __VA_ARGS__); \
      printf("%s\n", __RESET); \
		    } \
      } while (0)

#define DEBUG_PRINT_IF(statement, message) \
  do { \
    if ((statement)) { \
      printf("%s:%d:%s%s\n%sMESSAGE: %s%s\n", __FILE__,  __LINE__, __BLUE, \
          __func__, __GREEN, message, __RESET); \
		    } \
      } while (0)

#endif

#endif
