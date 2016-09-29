#if !defined(TENSOR_INCLUDE_TENSOR_H)
#define TENSOR_INCLUDE_TENSOR_H

#include "common_types.h"
#include "settings.h"

#if defined(ENABLE_PSI4)
#include <libmints/typedefs.h>
#include <ambit/helpers/psi4/integrals.h>
#endif

namespace ambit
{

// => Forward Declarations <=
class CompTensorImpl;
class LabeledCompTensor;
class LabeledCompTensorContraction;
class LabeledCompTensorAddition;
class LabeledCompTensorSubtraction;
class LabeledCompTensorDistribution;
class LabeledCompTensorSumOfProducts;
class SlicedCompTensor;

// => Tensor Types <=
enum CompTensorType
{
    CurrentTensor,     // <= If cloning from existing tensor use its type.
    CoreTensor        // <= In-core only tensor
    /* DiskTensor,        // <= Disk cachable tensor */
    /* DistributedTensor, // <= Tensor suitable for parallel distributed */
    /* AgnosticTensor     // <= Let the library decide for you. */
};

class CompTensor
{

  public:
    // => Constructors <= //

    /**
     * Factory constructor. Builds a Tensor of TensorType type
     * with given name and dimensions dims
     *
     * Parameters:
     *  @param type the TensorType to build
     *  @param name the name of the Tensor
     *  @param dims the dimensions of the indices of the tensor
     *    (dims.size() is the tensor rank)
     *
     * Results:
     *  @return new Tensor of TensorType type with name and dims
     *   The returned Tensor is set to zero.
     **/
    static CompTensor build(CompTensorType type, const string &name,
                        const Dimension &dims);

    /**
     * Return a new Tensor of TensorType type which copies the name,
     * dimensions, and data of this tensor.
     *
     * E.g.:
     *  Tensor A = C.clone(DiskTensor);
     * is equivalent to:
     *  Tensor A = Tensor::build(DiskTensor, C.name(), C.dims());
     *  A->copy(C);
     *
     * Parameters:
     *  @param type the TensorType to use for the clone
     *
     * Results:
     *  @return new Tensor of TensorType type with the name and contents of this
     **/
    CompTensor clone(CompTensorType type = CurrentTensor) const;

    /**
     * Default constructor, builds a Tensor with a null underlying
     * implementation.
     *
     * Calling any methods of such a Tensor will result in exceptions being
     * thrown.
     **/
    CompTensor();

    /**
     * Frees the Tensor's internal memory allocation. This is for users
     * that want finer control on memory consumption. After calling this
     * function the Tensor is no longer valid and cannot be used in
     * furthur calls.
     */
    void reset();

    // => Accessors <= //

    /// @return The tensor type enum, one of CoreTensor, DiskTensor,
    /// DistributedTensor
    CompTensorType type() const;
    /// @return The name of the tensor for use in printing
    string name() const;
    /// @return The dimension of each index in the tensor
    const Dimension &dims() const;
    /// @return The dimension of the ind-th index
    size_t dim(size_t ind) const;
    /// @return The number of indices in the tensor
    size_t rank() const;
    /// @return The total number of elements in the tensor (product of dims)
    size_t numel() const;

    /// Set the name of the tensor to name
    void set_name(const string &name);

    /// @return Does this Tensor point to the same underlying tensor as Tensor
    /// other?
    bool operator==(const CompTensor &other) const;
    /// @return !Does this Tensor point to the same underlying tensor as Tensor
    /// other?
    bool operator!=(const CompTensor &other) const;

    /**
     * Print some tensor information to fh
     * \param level If level = false, just print name and dimensions.  If level
     *= true, print the entire tensor.
     **/
    void print(FILE *fh = stdout, bool level = true,
               const string &format = string("%11.6f%+11.6fi"), int maxcols = 5) const;

    // => Data Access <= //

    /**
     * Returns the raw data vector underlying the tensor object if the
     * underlying tensor object supports a raw data vector. This is only the
     * case if the underlying tensor is of type CoreTensor.
     *
     * This routine is intended to facilitate rapid filling of data into a
     * CoreTensor buffer tensor, following which the user may stripe the buffer
     * tensor into a DiskTensor or DistributedTensor tensor via slice
     *operations.
     *
     * If a vector is successfully returned, it points to the unrolled data of
     * the tensor, with the right-most dimensions running fastest and left-most
     * dimensions running slowest.
     *
     * Example successful use case:
     *  Tensor A = Tensor::build(CoreTensor, "A3", {4,5,6});
     *  vector<double>& Av = A.data();
     *  double* Ap = Av.data(); // In case the raw pointer is needed
     *  In this case, Av[0] = A(0,0,0), Av[1] = A(0,0,1), etc.
     *
     *  Tensor B = Tensor::build(DiskTensor, "B3", {4,5,6});
     *  vector<double>& Bv = B.data(); // throws
     *
     * Results:
     *  @return data pointer, if tensor object supports it
     **/
    vector<complex<double>> &data();
    const vector<complex<double>> &data() const;

    // => BLAS-Type Tensor Operations <= //

    /**
     * Returns the norm of the tensor
     *
     * Parameters:
     * @param type the type of norm desired:
     *  0 - Infinity-norm, maximum absolute value of elements
     *  1 - One-norm, sum of absolute values of elements
     *  2 - Two-norm, square root of sum of squares
     *
     * Results:
     *  @return computed norm
     **/
    double norm(int type = 2) const;

    /** Find the maximum value.
     *
     * @return maximum value along with its indices
     */
    tuple<complex<double>, vector<size_t>> max() const;

    /** Find the minimum value.
     *
     * @return minimum value along with its indices
     */
    tuple<complex<double>, vector<size_t>> min() const;

    /**
     * Sets the data of the tensor to zeros
     *  C = 0.0
     *
     * Note: this just drops down to scale(0.0);
     *
     * Results:
     *  C is the current tensor, whose data is overwritten
     **/
    void zero();

    /**
     * Scales the tensor by scalar beta, e.g.:
     *  C = beta * C
     *
     * Note: If beta is 0.0, a memset is performed rather than a scale to clamp
     * NaNs and other garbage out.
     *
     * Parameters:
     *  @param beta the scale to apply
     *
     * Results:
     *  C is the current tensor, whose data is overwritten
     **/
    void scale(double beta = 0.0);
    void scale(complex<double> beta = 0.0);

    /**
     * Sets all elements in the tensor to the value.
     *
     * @param alpha the value to set
     */
    void set(complex<double> alpha);

    /**
     * Copy the data of other into this tensor:
     *  C() = other()
     * Note: this just drops into slice
     *
     * Parameters:
     *  @param other the tensor to copy data from
     *
     * Results
     *  C is the current tensor, whose data is overwritten
     **/
    void copy(const CompTensor &other);

    /**
     * Perform the slice:
     *  C(Cinds) = alpha * A(Ainds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2({{0,m},{0,n}}) += 0.5 * A2({{1,m+1},{1,n+1}});
     *
     * Parameters:
     *  @param A The source tensor, e.g., A2
     *  @param Cinds The slices of indices of tensor C, e.g., {{0,m},{0,n}}
     *  @param Ainds The indices of tensor A, e.g., {{1,m+1},{1,n+1}}
     *  @param alpha The scale applied to the tensor A, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     *  All elements outside of the IndexRange in C are untouched, alpha and
     *beta
     *  scales are applied only to elements indices of the IndexRange
     **/
    void slice(const CompTensor &A, const IndexRange &Cinds,
               const IndexRange &Ainds, complex<double> alpha = 1.0, complex<double> beta = 0.0);

    /**
     * Perform the permutation:
     *  C(Cinds) = alpha * A(Ainds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2("ij") += 0.5 * A2("ji");
     *
     * Parameters:
     *  @param A The source tensor, e.g., A2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ji"
     *  @param alpha The scale applied to the tensor A, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void permute(const CompTensor &A, const Indices &Cinds, const Indices &Ainds,
                 complex<double> alpha = 1.0, complex<double> beta = 0.0);

    /**
     * Perform the contraction:
     *  C(Cinds) = alpha * A(Ainds) * B(Binds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2("ij") += 0.5 * A2("ik") * B2("jk");
     *
     * Parameters:
     *  @param A The left-side factor tensor, e.g., A2
     *  @param B The right-side factor tensor, e.g., B2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ik"
     *  @param Binds The indices of tensor B, e.g., "jk"
     *  @param alpha The scale applied to the product A*B, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void contract(const CompTensor &A, const CompTensor &B, const Indices &Cinds,
                  const Indices &Ainds, const Indices &Binds,
                  complex<double> alpha = 1.0, complex<double> beta = 0.0);

    /**
     * Perform the GEMM call equivalent to:
     *  C_DGEMM(
     *      (transA ? 'T' : 'N'),
     *      (transB ? 'T' : 'N'),
     *      nrow,
     *      ncol,
     *      nzip,
     *      alpha,
     *      Ap + offA,
     *      ldaA,
     *      Bp + offB,
     *      ldaB,
     *      beta,
     *      Cp + offC,
     *      ldaC);
     *  where, e.g., Ap = A.data().data();
     *
     * Notes:
     *  - This is only implemented for CoreTensor
     *  - No bounds checking on the GEMM is performed
     *  - This function is intended to help advanced users get optimal
     *    performance from single-node codes.
     *
     * Parameters:
     *  @param A the left-side factor tensor
     *  @param B the right-side factor tensor
     *  @param transA transpose A or not
     *  @param transB transpose B or not
     *  @param nrow number of rows in the GEMM call
     *  @param ncol number of columns in the GEMM call
     *  @param nzip number of zip indices in the GEMM call
     *  @param ldaA leading dimension of A:
     *   Must be >= nzip if transA == false
     *   Must be >= nrow if transA == true
     *  @param ldaB leading dimension of B:
     *   Must be >= ncol if transB == false
     *   Must be >= nzip if transB == true
     *  @param ldaC leading dimension of C:
     *   Must be >= ncol
     *  @param offA the offset of the A data pointer to apply
     *  @param offB the offset of the B data pointer to apply
     *  @param offC the offset of the C data pointer to apply
     *  @param alpha the scale to apply to A*B
     *  @param beta the scale to apply to C
     *
     * Results:
     *  C is the current tensor, whose data is overwritten.
     *  All elements in C outside of the range traversed by gemm are
     *  untouched.
     **/
    void gemm(const CompTensor &A, const CompTensor &B, bool transA,bool conjA,
              bool transB, bool conjB,size_t nrow, size_t ncol, size_t nzip,
              size_t ldaA, size_t ldaB, size_t ldaC, size_t offA = 0L,
              size_t offB = 0L, size_t offC = 0L,
              double alpha = 1.0, double beta = 0.0);

    // => Rank-2 LAPACK-Type Tensor Operations <= //

    /**
     * This routine computes all the eigenvalues and eigenvectors of
     * a square real symmetric matrix (this, A).
     *
     * The eigenvector v(j) of this satisfies the following formula:
     *
     * A*v(j) = lambda(j)*v(j)
     *
     * where
     *
     * lambda(j) is its eigenvalue.
     *
     * The computed eigenvectors are orthonormal.
     *
     * @returns map of Tensor with the keys "eigenvalues" and "eigenvectors".
     */
    map<string, CompTensor> heev(EigenvalueOrder order) const;
    CompTensor power(double power, double condition = 1.0E-12) const;

    /**
     * This routine computes a square real general matrix (this, A), the
     * eigenvalues and the left and right eigenvectors.
     *
     * The right eigenvector v(j) of A satisfies the following formula:
     *
     * A*v(j) = lambda(j)*v(j)
     *
     * where
     *
     * lambda(j) is its eigenvalue.
     *
     * The left eigenvector u(j) of A satisfies the following formula:
     *
     * u(j)H*A = lambda(j)*u(j)H
     *
     * where
     *
     * u(j)H denotes the conjugate transpose of u(j).
     *
     * The computed eigenvectors are normalized so that their Euclidean
     * norm equals one and the largest component is real.
     *
     * @returns map of Tensor with the keys "lambda", "lambda i", "v", and "u".
     * See definitions above.
     */

    // void potrf();
    // void potri();
    // void potrs(const Tensor& L);
    // void posv(const Tensor& A);

    // void trtrs(const Tensor& L,

    // void getrf();
    // void getri();
    // void getrs(const Tensor& LU);
    // void gesv(const Tensor& A);

    // map<string, Tensor> lu() const;
    // map<string, Tensor> qr() const;

    CompTensor inverse() const;

    // => Utility Operations <= //

    static CompTensor cat(const vector<CompTensor>, int dim);

    // => Iterators <= //

    void iterate(const function<void(const vector<size_t> &, complex<double> &)> &func);
    void citerate(const function<void(const vector<size_t> &, const complex<double> &)>
                      &func) const;

  private:
  protected:
    shared_ptr<CompTensorImpl> tensor_;

    CompTensor(shared_ptr<CompTensorImpl> tensor);

    static map<string, CompTensor>
    map_to_tensor(const map<string, CompTensorImpl *> &x);

    void reshape(const Dimension &dims);

  public:
    // => Operator Overloading API <= //

    LabeledCompTensor operator()(const string &indices) const;

    SlicedCompTensor operator()(const IndexRange &range) const;
    SlicedCompTensor operator()() const;

    // => Environment <= //

  private:
    static string scratch_path__;

  public:
    static void set_scratch_path(const string &path) { scratch_path__ = path; }
    static string scratch_path() { return scratch_path__; }

#if defined(ENABLE_PSI4)
    friend void ambit::helpers::psi4::integrals(psi::TwoBodyAOInt &integral,
                                                ambit::Tensor *target);
#endif
};

class LabeledCompTensor
{
  public:
    LabeledCompTensor(CompTensor T, const Indices &indices, complex<double> factor = 1.0);

    complex<double> factor() const { return factor_; }
    const Indices &indices() const { return indices_; }
    Indices &indices() { return indices_; }
    const Tensor &T() const { return T_; }

    LabeledCompTensorContraction operator*(const LabeledCompTensor &rhs);
    LabeledCompTensorAddition operator+(const LabeledCompTensor &rhs);
    LabeledCompTensorAddition operator-(const LabeledCompTensor &rhs);

    LabeledCompTensorDistribution operator*(const LabeledCompTensorAddition &rhs);

    /** Copies data from rhs to this sorting the data if needed. */
    void operator=(const LabeledCompTensor &rhs);
    void operator+=(const LabeledCompTensor &rhs);
    void operator-=(const LabeledCompTensor &rhs);

    void operator=(const LabeledCompTensorDistribution &rhs);
    void operator+=(const LabeledCompTensorDistribution &rhs);
    void operator-=(const LabeledCompTensorDistribution &rhs);

    void operator=(const LabeledCompTensorContraction &rhs);
    void operator+=(const LabeledCompTensorContraction &rhs);
    void operator-=(const LabeledCompTensorContraction &rhs);

    void operator=(const LabeledCompTensorAddition &rhs);
    void operator+=(const LabeledCompTensorAddition &rhs);
    void operator-=(const LabeledCompTensorAddition &rhs);

    void operator*=(complex<double> scale);
    void operator/=(complex<double> scale);

    //    bool operator==(const LabeledTensor& other) const;
    //    bool operator!=(const LabeledTensor& other) const;

    size_t numdim() const { return indices_.size(); }
    size_t dim_by_index(const string &idx) const;

    // negation
    LabeledCompTensor operator-() const
    {
        return LabeledCompTensor(T_, indices_, -factor_);
    }

    void contract(const LabeledCompTensorContraction &rhs, bool zero_result,
                  bool add);

  private:
    void set(const LabeledCompTensor &to);

    CompTensor T_;
    Indices indices_;
    complex<double> factor_;
};

inline LabeledCompTensor operator*(complex<double> factor, const LabeledCompTensor &ti)
{
    return LabeledCompTensor(ti.T(), ti.indices(), factor * ti.factor());
};

class LabeledCompTensorContraction
{

  public:
    LabeledCompTensorContraction(const LabeledCompTensor &A, const LabeledCompTensor &B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    LabeledTensorContraction() {}

    size_t size() const { return tensors_.size(); }

    const LabeledCompTensor &operator[](size_t i) const { return tensors_[i]; }

    LabeledCompTensorContraction &operator*(const LabeledCompTensor &other)
    {
        tensors_.push_back(other);
        return *this;
    }

    void operator*=(const LabeledCompTensor &other) { tensors_.push_back(other); }

    // conversion operator
    operator complex<double>() const;

    pair<double, double>
    compute_contraction_cost(const vector<size_t> &perm) const;

  private:
    vector<LabeledCompTensor> tensors_;
};

class LabeledCompTensorAddition
{
  public:
    LabeledCompTensorAddition(const LabeledCompTensor &A, const LabeledCompTensor &B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledCompTensor &operator[](size_t i) const { return tensors_[i]; }

    vector<LabeledCompTensor>::iterator begin() { return tensors_.begin(); }
    vector<LabeledCompTensor>::const_iterator begin() const
    {
        return tensors_.begin();
    }

    vector<LabeledCompTensor>::iterator end() { return tensors_.end(); }
    vector<LabeledCompTensor>::const_iterator end() const { return tensors_.end(); }

    LabeledTensorCompAddition &operator+(const LabeledCompTensor &other)
    {
        tensors_.push_back(other);
        return *this;
    }

    LabeledCompTensorAddition &operator-(const LabeledCompTensor &other)
    {
        tensors_.push_back(-other);
        return *this;
    }

    LabeledCompTensorDistribution operator*(const LabeledCompTensor &other);

    LabeledCompTensorAddition &operator*(complex<double> scalar);

    // negation
    LabeledCompTensorAddition &operator-();

  private:
    // This handles cases like T("ijab")
    vector<LabeledCompTensor> tensors_;
};

inline LabeledCompTensorAddition operator*(complex<double> factor,
                                       const LabeledCompTensorAddition &ti)
{
    LabeledCompTensorAddition ti2 = ti;
    return ti2 * factor;
}

// Is responsible for expressions like D * (J - K) --> D*J - D*K
class LabeledCompTensorDistribution
{
  public:
    LabeledCompTensorDistribution(const LabeledCompTensor &A,
                              const LabeledCompTensorAddition &B)
        : A_(A), B_(B)
    {
    }

    const LabeledCompTensor &A() const { return A_; }
    const LabeledCompTensorAddition &B() const { return B_; }

    // conversion operator
    operator complex<double>() const;

  private:
    const LabeledCompTensor &A_;
    const LabeledCompTensorAddition &B_;
};

class SlicedCompTensor
{
  public:
    SlicedCompTensor(CompTensor T, const IndexRange &range, complex<double> factor = 1.0);

    complex<double> factor() const { return factor_; }
    const IndexRange &range() const { return range_; }
    const CompTensor &T() const { return T_; }

    void operator=(const SlicedCompTensor &rhs);
    void operator+=(const SlicedCompTensor &rhs);
    void operator-=(const SlicedCompTensor &rhs);

    // negation
    SlicedCompTensor operator-() const
    {
        return SlicedCompTensor(T_, range_, -factor_);
    }

  private:
    CompTensor T_;
    IndexRange range_;
    complex<double> factor_;
};

inline SlicedCompTensor operator*(complex<double> factor, const SlicedCompTensor &ti)
{
    return SlicedCompTensor(ti.T(), ti.range(), factor * ti.factor());
};
}

#endif
