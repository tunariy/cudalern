#include <cudalern/Containers/NdArray.hpp>
#include <cudalern/Core/Device/device.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// -----------------------------------------------------------------------------
// Test fixture – initializes CUDA context once for all tests
// -----------------------------------------------------------------------------
class NdArrayTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() { cudalern::internal::InitializeContext(0); }
};

// -----------------------------------------------------------------------------
// Scalar arithmetic tests (new array)
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, ScalarAdd) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto result = arr + 3;
    auto host = result.data();
    std::vector<int> expected = {8, 8, 8, 8};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, ScalarSub) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto result = arr - 3;
    auto host = result.data();
    std::vector<int> expected = {2, 2, 2, 2};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, ScalarMul) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto result = arr * 3;
    auto host = result.data();
    std::vector<int> expected = {15, 15, 15, 15};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, ScalarDiv) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 10);
    auto result = arr / 2;
    auto host = result.data();
    std::vector<int> expected = {5, 5, 5, 5};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// Array-array elementwise arithmetic tests (new array)
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, ArrayAdd) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto C = A + B;
    auto host = C.data();
    std::vector<int> expected = {8, 8, 8, 8};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, ArraySub) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto C = A - B;
    auto host = C.data();
    std::vector<int> expected = {2, 2, 2, 2};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, ArrayMulElementwise) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto C = A % B;  // elementwise for rank=2 (matrix multiplication is for rank>=2 but
                     // overloaded)
    auto host = C.data();
    std::vector<int> expected = {15, 15, 15, 15};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, ArrayDiv) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 10);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto C = A / B;
    auto host = C.data();
    std::vector<int> expected = {5, 5, 5, 5};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// In-place scalar arithmetic
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, InplaceScalarAdd) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    arr += 3;
    auto host = arr.data();
    std::vector<int> expected = {8, 8, 8, 8};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, InplaceScalarSub) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    arr -= 3;
    auto host = arr.data();
    std::vector<int> expected = {2, 2, 2, 2};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, InplaceScalarMul) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    arr *= 3;
    auto host = arr.data();
    std::vector<int> expected = {15, 15, 15, 15};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, InplaceScalarDiv) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 10);
    arr /= 2;
    auto host = arr.data();
    std::vector<int> expected = {5, 5, 5, 5};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// In-place array-array arithmetic
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, InplaceArrayAdd) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    A += B;
    auto host = A.data();
    std::vector<int> expected = {8, 8, 8, 8};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, InplaceArraySub) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    A -= B;
    auto host = A.data();
    std::vector<int> expected = {2, 2, 2, 2};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, InplaceArrayMul) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    A *= B;
    auto host = A.data();
    std::vector<int> expected = {15, 15, 15, 15};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, InplaceArrayDiv) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 10);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    A /= B;
    auto host = A.data();
    std::vector<int> expected = {5, 5, 5, 5};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// Unary negation
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, UnaryNegate) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 5);
    auto result = -arr;
    auto host = result.data();
    std::vector<int> expected = {-5, -5, -5, -5};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// Comparison operators (return 0/1 arrays)
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, CompareEqual) {
    auto A = cudalern::NdArray<int, 2>::ones({2, 2});
    auto B = cudalern::NdArray<int, 2>::ones({2, 2});
    auto C = (A == B);
    auto host = C.data();
    std::vector<int> expected = {1, 1, 1, 1};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, CompareNotEqual) {
    auto A = cudalern::NdArray<int, 2>::ones({2, 2});
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto C = (A != B);
    auto host = C.data();
    std::vector<int> expected = {1, 1, 1, 1};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, CompareLess) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto C = (A < B);
    auto host = C.data();
    std::vector<int> expected = {1, 1, 1, 1};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, CompareGreater) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto C = (A > B);
    auto host = C.data();
    std::vector<int> expected = {1, 1, 1, 1};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, CompareLessEqual) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto C = (A <= B);
    auto host = C.data();
    std::vector<int> expected = {1, 1, 1, 1};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, CompareGreaterEqual) {
    auto A = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto B = cudalern::NdArray<int, 2>::full({2, 2}, 3);
    auto C = (A >= B);
    auto host = C.data();
    std::vector<int> expected = {1, 1, 1, 1};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// Math functions
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, MathPow) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, 2);
    auto result = arr.pow(3);
    auto host = result.data();
    std::vector<int> expected = {8, 8, 8, 8};
    EXPECT_EQ(host, expected);
}

TEST_F(NdArrayTest, MathExp) {
    auto arr = cudalern::NdArray<float, 2>::full({2, 2}, 0.0f);
    auto result = arr.exp();
    auto host = result.data();
    for (float v : host) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

TEST_F(NdArrayTest, MathLog) {
    auto arr = cudalern::NdArray<float, 2>::full({2, 2}, std::exp(1.0f));
    auto result = arr.log();
    auto host = result.data();
    for (float v : host) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

TEST_F(NdArrayTest, MathSqrt) {
    auto arr = cudalern::NdArray<float, 2>::full({2, 2}, 4.0f);
    auto result = arr.sqrt();
    auto host = result.data();
    for (float v : host) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }
}

TEST_F(NdArrayTest, MathAbs) {
    auto arr = cudalern::NdArray<int, 2>::full({2, 2}, -5);
    auto result = arr.abs();
    auto host = result.data();
    std::vector<int> expected = {5, 5, 5, 5};
    EXPECT_EQ(host, expected);
}

// -----------------------------------------------------------------------------
// Activation functions
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, ActivationRelu) {
    std::vector<std::vector<float>> data = {{-1.0f, 0.5f}, {2.0f, -0.5f}};
    auto arr = cudalern::NdArray<float, 2>(data);
    auto result = arr.relu();
    auto host = result.data();
    std::vector<float> expected = {0.0f, 0.5f, 2.0f, 0.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

TEST_F(NdArrayTest, ActivationLeakyRelu) {
    std::vector<std::vector<float>> data = {{-1.0f, 0.5f}, {2.0f, -0.5f}};
    auto arr = cudalern::NdArray<float, 2>(data);
    auto result = arr.leaky_relu(0.1f);
    auto host = result.data();
    std::vector<float> expected = {-0.1f, 0.5f, 2.0f, -0.05f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

TEST_F(NdArrayTest, ActivationSigmoid) {
    auto arr = cudalern::NdArray<float, 2>::full({2, 2}, 0.0f);
    auto result = arr.sigmoid();
    auto host = result.data();
    for (float v : host) {
        EXPECT_FLOAT_EQ(v, 0.5f);
    }
}

TEST_F(NdArrayTest, ActivationTanh) {
    auto arr = cudalern::NdArray<float, 2>::full({2, 2}, 0.0f);
    auto result = arr.tanh();
    auto host = result.data();
    for (float v : host) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

// -----------------------------------------------------------------------------
// Elementwise multiply for Rank=1 (ensure no conflict with matrix multiply)
// -----------------------------------------------------------------------------

TEST_F(NdArrayTest, ElementwiseMulRank1) {
    auto A = cudalern::NdArray<int, 1>::full({3}, 2);
    auto B = cudalern::NdArray<int, 1>::full({3}, 3);
    auto C = A * B;  // should be elementwise, not matrix multiply (rank=1)
    auto host = C.data();
    std::vector<int> expected = {6, 6, 6};
    EXPECT_EQ(host, expected);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}