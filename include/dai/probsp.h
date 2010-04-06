/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  libDAI is licensed under the terms of the GNU General Public License version
 *  2, or (at your option) any later version. libDAI is distributed without any
 *  warranty. See the file COPYING for more details.
 *
 *  Copyright (C) 2006-2009  Joris Mooij  [joris dot mooij at libdai dot org]
 *  Copyright (C) 2006-2007  Radboud University Nijmegen, The Netherlands
 */


/// \file
/// \brief Defines TProbSp<> class which represents sparse (probability) vectors


#ifndef __defined_libdai_probsp_h
#define __defined_libdai_probsp_h


#include <cmath>
#include <vector>
#include <ostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <dai/util.h>
#include <dai/exceptions.h>
#include <dai/fo.h>


namespace dai {


/// Represents a vector with entries of type \a T.
/** It is simply a <tt>std::vector</tt><<em>T</em>> with an interface designed for dealing with probability mass functions.
 *
 *  It is mainly used for representing measures on a finite outcome space, for example, the probability
 *  distribution of a discrete random variable. However, entries are not necessarily non-negative; it is also used to
 *  represent logarithms of probability mass functions.
 *
 *  \tparam T Should be a scalar that is castable from and to dai::Real and should support elementary arithmetic operations.
 */
template <typename T, typename spvector_type>
class TProbSp {
    public:
        /// Type of data structure used for storing the values
        typedef spvector_type container_type;
        typedef TProbSp<T,spvector_type> this_type;

    private:
        /// The data structure that stores the nondefault values
        container_type _p;

    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor (constructs empty vector)
        TProbSp() : _p() {}

        /// Construct uniform probability distribution over \a n outcomes (i.e., a vector of length \a n with each entry set to \f$1/n\f$)
        explicit TProbSp( size_t n ) : _p( n, (T)1 / n ) {}

        /// Construct vector of length \a n with each entry set to \a p
        explicit TProbSp( size_t n, T p ) : _p( n, p ) {}

        /// Construct sparse vector from a range
        /** \tparam TIterator Iterates over instances that can be cast to \a T
         *  \param begin Points to first instance to be added.
         *  \param end Points just beyond last instance to be added.
         *  \param sizeHint For efficiency, the number of entries can be speficied by \a sizeHint;
         *    the value 0 can be given if the size is unknown, but this will result in a performance penalty.
         *  \param def Default value to use for the constructed sparse vector.
         *  \note In libDAI versions 0.2.4 and earlier, the \a sizeHint argument was optional.
         */
        template <typename TIterator>
        TProbSp( TIterator begin, TIterator end, size_t sizeHint, T def=0 ) : _p( begin, end, sizeHint, def ) {}

        /// Construct vector from another vector
        /** \tparam S type of elements in \a v (should be castable to type \a T)
         *  \param v vector used for initialization.
         *  \param def Default value to used for the constructed sparse vector.
         */
        template <typename S>
        TProbSp( const std::vector<S> &v, T def=0 ) : _p( v, v.size(), def ) {}
    //@}

        /// Constant iterator over the elements
        typedef typename container_type::const_iterator const_iterator;
        /// Iterator over the elements
        typedef typename container_type::iterator iterator;
        /// Constant reverse iterator over the elements
        typedef typename container_type::const_reverse_iterator const_reverse_iterator;
        /// Reverse iterator over the elements
        typedef typename container_type::reverse_iterator reverse_iterator;

    /// \name Iterator interface
    //@{
        /// Returns iterator that points to the first nondefault element
        iterator begin() { return _p.begin(); }
        /// Returns constant iterator that points to the first nondefault element
        const_iterator begin() const { return _p.begin(); }

        /// Returns iterator that points beyond the last nondefault element
        iterator end() { return _p.end(); }
        /// Returns constant iterator that points beyond the last nondefault element
        const_iterator end() const { return _p.end(); }

        /// Returns reverse iterator that points to the last nondefault element
        reverse_iterator rbegin() { return _p.rbegin(); }
        /// Returns constant reverse iterator that points to the last nondefault element
        const_reverse_iterator rbegin() const { return _p.rbegin(); }

        /// Returns reverse iterator that points beyond the first nondefault element
        reverse_iterator rend() { return _p.rend(); }
        /// Returns constant reverse iterator that points beyond the first nondefault element
        const_reverse_iterator rend() const { return _p.rend(); }
    //@}

    /// \name Miscellaneous operations
    //@{
        void resize( size_t sz ) {
            _p.resize( sz );
        }
    //@}

    /// \name Get/set individual entries
    //@{
        /// Gets \a i 'th entry
        T get( size_t i ) const { return _p[i]; }

        /// Sets \a i 'th entry to \a val
        void set( size_t i, T val ) { _p.set( i, val ); }
    //@}

    /// \name Queries
    //@{
        /// Returns a const reference to the wrapped container
        const container_type& p() const { return _p; }

        /// Returns a reference to the wrapped container
        container_type& p() { return _p; }

        /// Returns a copy of the \a i 'th entry
        T operator[]( size_t i ) const { return get(i); }

        /// Returns length of the vector (i.e., the number of entries)
        size_t size() const { return _p.size(); }

        /// Returns number of default values
        size_t nrDef() const { return _p.nrDef(); }

        /// Returns number of nondefault values
        size_t nrNonDef() const { return _p.nrNonDef(); }

        /// Returns default value
        T def() const { return _p.def(); }

        /// Sets default value
        void setDef( T def ) { _p.setDef( def ); }

        /// Accumulate all values (similar to std::accumulate) by summing
        /** The following calculation is done:
         *  \code
         *  T t = op(init);
         *  for( const_iterator it = begin(); it != end(); it++ )
         *      t += op(*it);
         *  return t;
         *  \endcode
         */
        template<typename unOp> T accumulateSum( T init, unOp op ) const {
            T t = op(init);
            for( const_iterator it = begin(); it != end(); it++ )
                t += op(it->second);
            t += nrDef() * op(def());
            return t;
        }

        /// Accumulate all values (similar to std::accumulate) by maximization/minimization
        /** The following calculation is done (with "max" replaced by "min" if \a minimize == \c true):
         *  \code
         *  T t = op(init);
         *  for( const_iterator it = begin(); it != end(); it++ )
         *      t = std::max( t, op(*it) );
         *  return t;
         *  \endcode
         */
        template<typename unOp> T accumulateMax( T init, unOp op, bool minimize ) const {
            T t = op(init);
            if( minimize ) {
                for( const_iterator it = begin(); it != end(); it++ )
                    t = std::min( t, op(it->second) );
                if( nrDef() )
                    t = std::min( t, op(def()) );
            } else {
                for( const_iterator it = begin(); it != end(); it++ )
                    t = std::max( t, op(it->second) );
                if( nrDef() )
                    t = std::max( t, op(def()) );
            }
            return t;
        }

        /// Returns the Shannon entropy of \c *this, \f$-\sum_i p_i \log p_i\f$
        T entropy() const { return -accumulateSum( (T)0, fo_plog0p<T>() ); }

        /// Returns maximum value of all entries
        T max() const { return accumulateMax( (T)(-INFINITY), fo_id<T>(), false ); }

        /// Returns minimum value of all entries
        T min() const { return accumulateMax( (T)INFINITY, fo_id<T>(), true ); }

        /// Returns sum of all entries
        T sum() const { return accumulateSum( (T)0, fo_id<T>() ); }

        /// Return sum of absolute value of all entries
        T sumAbs() const { return accumulateSum( (T)0, fo_abs<T>() ); }

        /// Returns maximum absolute value of all entries
        T maxAbs() const { return accumulateMax( (T)0, fo_abs<T>(), false ); }

        /// Returns \c true if one or more entries are NaN
        bool hasNaNs() const {
            if( isnan( def() ) && nrDef() )
                return true;
            else {
                bool foundnan = false;
                for( const_iterator it = begin(); it != end(); it++ )
                    if( isnan( it->second ) ) {
                        foundnan = true;
                        break;
                    }
                return foundnan;
            }
        }

        /// Returns \c true if one or more entries are negative
        bool hasNegatives() const {
            if( (def() < 0) && nrDef() )
                return true;
            else {
                bool foundnegative = false;
                for( const_iterator it = begin(); it != end(); it++ )
                    if( it->second < 0 ) {
                        foundnegative = true;
                        break;
                    }
                return foundnegative;
            }
        }

        /// Returns a pair consisting of the index of the maximum value and the maximum value itself
        std::pair<size_t,T> argmax() const {
            T max;
            size_t arg;
            DAI_ASSERT( size() );
            if( nrDef() == size() ) {
                max = def();
                arg = 0;
            } else if( nrDef() > 0 ) {
                max = begin()->second;
                arg = begin()->first;
                size_t i = 0;
                size_t argdef = 0;
                for( const_iterator it = begin(); it != end(); it++ ) {
                    if( it->second > max ) {
                        max = it->second;
                        arg = it->first;
                    }
                    if( it->first != i )
                        argdef = i;
                    i = it->first + 1;
                }
                if( def() > max ) {
                    max = def();
                    arg = argdef;
                }
            } else {
                max = begin()->second;
                arg = begin()->first;
                for( const_iterator it = begin(); it != end(); it++ )
                    if( it->second > max ) {
                        max = it->second;
                        arg = it->first;
                    }
            }
            return std::make_pair( arg, max );
        }

        /// Returns a random index, according to the (normalized) distribution described by *this
        size_t draw() {
            Real x = rnd_uniform() * sum();
            T s = 0;
            for( size_t i = 0; i < size(); i++ ) {
                s += get(i);
                if( s > x )
                    return i;
            }
            return( size() - 1 );
        }

        /// Lexicographical comparison
        /** \pre <tt>this->size() == q.size()</tt>
         */
        bool operator<( const this_type& q ) const {
            DAI_DEBASSERT( size() == q.size() );
            for( size_t i = 0; i < size(); i++ ) {
                T a = get(i);
                T b = q.get(i);
                if( a > b )
                    return false;
                if( a < b )
                    return true;
            }
            return false;
        }

        /// Comparison
        bool operator==( const this_type& q ) const {
            return _p == q._p; 
        }
    //@}

    /// \name Unary transformations
    //@{
        /// Returns the result of applying operation \a op pointwise on \c *this
        template<typename unaryOp> this_type pwUnaryTr( unaryOp op ) const {
            this_type r;
            r.setDef( op( def() ) );
            r._p.resize( size() );
            for( const_iterator it = begin(); it != end(); it++ ) {
                T new_val = op( it->second );
                if( new_val != r.def() )
                    r._p.push_back( it->first, new_val );
            }
            return r;
        }

        /// Returns negative of \c *this
        this_type operator- () const { return pwUnaryTr( std::negate<T>() ); }

        /// Returns pointwise absolute value
        this_type abs() const { return pwUnaryTr( fo_abs<T>() ); }

        /// Returns pointwise exponent
        this_type exp() const { return pwUnaryTr( fo_exp<T>() ); }

        /// Returns pointwise logarithm
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        this_type log(bool zero=false) const {
            if( zero )
                return pwUnaryTr( fo_log0<T>() );
            else
                return pwUnaryTr( fo_log<T>() );
        }

        /// Returns pointwise inverse
        /** If \a zero == \c true, uses <tt>1/0==0</tt>; otherwise, <tt>1/0==Inf</tt>.
         */
        this_type inverse(bool zero=true) const {
            if( zero )
                return pwUnaryTr( fo_inv0<T>() );
            else
                return pwUnaryTr( fo_inv<T>() );
        }

        /// Returns normalized copy of \c *this, using the specified norm
        /** \throw NOT_NORMALIZABLE if the norm is zero
         */
        this_type normalized( ProbNormType norm = NORMPROB ) const {
            T Z = 0;
            if( norm == NORMPROB )
                Z = sum();
            else if( norm == NORMLINF )
                Z = maxAbs();
            if( Z == (T)0 ) {
                DAI_THROW(NOT_NORMALIZABLE);
                return *this;
            } else
                return pwUnaryTr( std::bind2nd( std::divides<T>(), Z ) );
        }
    //@}

    /// \name Unary operations
    //@{
        /// Applies unary operation \a op pointwise
        template<typename unaryOp> this_type& pwUnaryOp( unaryOp op ) {
            setDef( op( def() ) );
            for( iterator it = begin(); it != end(); ) {
                T new_val = op( it->second );
                if( new_val != def() ) {
                    it->second = new_val;
                    it++;
                } else
                    it = _p.erase( it );
            }
            return *this;
        }

        /// Draws all entries i.i.d. from a uniform distribution on [0,1)
        this_type& randomize() {
            setDef( 0 );
            for( size_t i = 0; i < size(); i++ )
                set( i, (T)rnd_uniform() );
            return *this;
        }

        /// Sets all entries to \f$1/n\f$ where \a n is the length of the vector
        this_type& setUniform () {
            setDef( (T)1 / size() );
            _p.clearNonDef();
            return *this;
        }

        /// Applies absolute value pointwise
        this_type& takeAbs() { return pwUnaryOp( fo_abs<T>() ); }

        /// Applies exponent pointwise
        this_type& takeExp() { return pwUnaryOp( fo_exp<T>() ); }

        /// Applies logarithm pointwise
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        this_type& takeLog(bool zero=false) {
            if( zero ) {
                return pwUnaryOp( fo_log0<T>() );
            } else
                return pwUnaryOp( fo_log<T>() );
        }

        /// Normalizes vector using the specified norm
        /** \throw NOT_NORMALIZABLE if the norm is zero
         */
        T normalize( ProbNormType norm=NORMPROB ) {
            T Z = 0;
            if( norm == NORMPROB )
                Z = sum();
            else if( norm == NORMLINF )
                Z = maxAbs();
            if( Z == (T)0 )
                DAI_THROW(NOT_NORMALIZABLE);
            else
                *this /= Z;
            return Z;
        }
    //@}

    /// \name Operations with scalars
    //@{
        /// Sets all entries to \a x
        this_type& fill( T x ) {
            setDef( x );
            _p.clearNonDef();
            return *this;
        }

        /// Adds scalar \a x to each entry
        this_type& operator+= (T x) {
            if( x != 0 )
                return pwUnaryOp( std::bind2nd( std::plus<T>(), x ) );
            else
                return *this;
        }

        /// Subtracts scalar \a x from each entry
        this_type& operator-= (T x) {
            if( x != 0 )
                return pwUnaryOp( std::bind2nd( std::minus<T>(), x ) );
            else
                return *this;
        }

        /// Multiplies each entry with scalar \a x
        this_type& operator*= (T x) {
            if( x != 1 )
                return pwUnaryOp( std::bind2nd( std::multiplies<T>(), x ) );
            else
                return *this;
        }

        /// Divides each entry by scalar \a x, where division by 0 yields 0
        this_type& operator/= (T x) {
            if( x != 1 )
                return pwUnaryOp( std::bind2nd( fo_divides0<T>(), x ) );
            else
                return *this;
        }

        /// Raises entries to the power \a x
        this_type& operator^= (T x) {
            if( x != (T)1 )
                return pwUnaryOp( std::bind2nd( fo_pow<T>(), x) );
            else
                return *this;
        }
    //@}

    /// \name Transformations with scalars
    //@{
        /// Returns sum of \c *this and scalar \a x
        this_type operator+ (T x) const { return pwUnaryTr( std::bind2nd( std::plus<T>(), x ) ); }

        /// Returns difference of \c *this and scalar \a x
        this_type operator- (T x) const { return pwUnaryTr( std::bind2nd( std::minus<T>(), x ) ); }

        /// Returns product of \c *this with scalar \a x
        this_type operator* (T x) const { return pwUnaryTr( std::bind2nd( std::multiplies<T>(), x ) ); }

        /// Returns quotient of \c *this and scalar \a x, where division by 0 yields 0
        this_type operator/ (T x) const { return pwUnaryTr( std::bind2nd( fo_divides0<T>(), x ) ); }

        /// Returns \c *this raised to the power \a x
        this_type operator^ (T x) const { return pwUnaryTr( std::bind2nd( fo_pow<T>(), x ) ); }
    //@}

    /// \name Operations with other equally-sized vectors
    //@{
        /// Applies binary operation pointwise on two vectors
        /** \tparam binaryOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param q Right operand
         *  \param op Operation of type \a binaryOp
         */
        template<typename binaryOp> this_type& pwBinaryOp( const this_type &q, binaryOp op ) {
            DAI_DEBASSERT( size() == q.size() );
            this_type p(*this);
            setDef( op( p.def(), q.def() ) );
            for( typename this_type::const_iterator it = p.begin(); it != p.end(); it++ ) {
                T new_val = op( it->second, q[it->first] );
                set( it->first, new_val );
            }
            for( typename this_type::const_iterator it = q.begin(); it != q.end(); it++ ) {
                T new_val = op( p[it->first], it->second );
                set( it->first, new_val );
            }
            return *this;
        }

        /// Pointwise addition with \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type& operator+= (const this_type & q) { return pwBinaryOp( q, std::plus<T>() ); }

        /// Pointwise subtraction of \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type& operator-= (const this_type & q) { return pwBinaryOp( q, std::minus<T>() ); }

        /// Pointwise multiplication with \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type& operator*= (const this_type & q) { return pwBinaryOp( q, std::multiplies<T>() ); }

        /// Pointwise division by \a q, where division by 0 yields 0
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see divide(const this_type &)
         */
        this_type& operator/= (const this_type & q) { return pwBinaryOp( q, fo_divides0<T>() ); }

        /// Pointwise division by \a q, where division by 0 yields +Inf
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see operator/=(const this_type &)
         */
        this_type& divide (const this_type & q) { return pwBinaryOp( q, std::divides<T>() ); }

        /// Pointwise power
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type& operator^= (const this_type & q) { return pwBinaryOp( q, fo_pow<T>() ); }
    //@}

    /// \name Transformations with other equally-sized vectors
    //@{
        /// Returns the result of applying binary operation \a op pointwise on \c *this and \a q
        /** \tparam binaryOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param q Right operand
         *  \param op Operation of type \a binaryOp
         */
        template<typename binaryOp> this_type pwBinaryTr( const this_type &q, binaryOp op ) const {
            DAI_DEBASSERT( size() == q.size() );
            this_type result;
            result.setDef( op( def(), q.def() ) );
            result._p.resize( size() );
            for( typename this_type::const_iterator it = begin(); it != end(); it++ ) {
                T new_val = op( it->second, q[it->first] );
                if( new_val != result.def() )
                    result._p.push_back(it->first, new_val);
            }
            for( typename this_type::const_iterator it = q.begin(); it != q.end(); it++ ) {
                T new_val = op( get(it->first), it->second );
                if( new_val != result.def() )
                    result.set( it->first, new_val );
            }
            return result;
        }

        /// Returns sum of \c *this and \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type operator+ ( const this_type& q ) const { return pwBinaryTr( q, std::plus<T>() ); }

        /// Return \c *this minus \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type operator- ( const this_type& q ) const { return pwBinaryTr( q, std::minus<T>() ); }

        /// Return product of \c *this with \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type operator* ( const this_type &q ) const { return pwBinaryTr( q, std::multiplies<T>() ); }

        /// Returns quotient of \c *this with \a q, where division by 0 yields 0
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see divided_by(const this_type &)
         */
        this_type operator/ ( const this_type &q ) const { return pwBinaryTr( q, fo_divides0<T>() ); }

        /// Pointwise division by \a q, where division by 0 yields +Inf
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see operator/(const this_type &)
         */
        this_type divided_by( const this_type &q ) const { return pwBinaryTr( q, std::divides<T>() ); }

        /// Returns \c *this to the power \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        this_type operator^ ( const this_type &q ) const { return pwBinaryTr( q, fo_pow<T>() ); }
    //@}

        /// Performs a generalized inner product, similar to std::inner_product
        /** \pre <tt>this->size() == q.size()</tt>
         */
        template<typename binOp1, typename binOp2> T innerProduct( const this_type &q, T init, binOp1 binaryOp1, binOp2 binaryOp2 ) const {
            DAI_DEBASSERT( size() == q.size() );
            // OPTIMIZE ME
            T result = init;
            for( size_t i = 0; i < size(); i++ )
                result = binaryOp1( result, binaryOp2( get(i), q.get(i) ) );
            return result;
        }
};


/// Returns distance between \a p and \a q, measured using distance measure \a dt
/** \relates TProbSp
 *  \pre <tt>this->size() == q.size()</tt>
 */
template<typename T, typename spvector_type> T dist( const TProbSp<T,spvector_type>& p, const TProbSp<T,spvector_type>& q, ProbDistType dt ) {
    switch( dt ) {
        case DISTL1:
            return p.innerProduct( q, (T)0, std::plus<T>(), fo_absdiff<T>() );
        case DISTLINF:
            return p.innerProduct( q, (T)0, fo_max<T>(), fo_absdiff<T>() );
        case DISTTV:
            return p.innerProduct( q, (T)0, std::plus<T>(), fo_absdiff<T>() ) / 2;
        case DISTKL:
            return p.innerProduct( q, (T)0, std::plus<T>(), fo_KL<T>() );
        case DISTHEL:
            return p.innerProduct( q, (T)0, std::plus<T>(), fo_Hellinger<T>() ) / 2;
        default:
            DAI_THROW(UNKNOWN_ENUM_VALUE);
            return INFINITY;
    }
}


/// Writes a TProbSp to an output stream
/** \relates TProbSp
 */
template<typename T, typename spvector_type> std::ostream& operator<< (std::ostream& os, const TProbSp<T,spvector_type>& p) {
    os << p.p();
    return os;
}


/// Returns the pointwise minimum of \a a and \a b
/** \relates TProbSp
 *  \pre <tt>this->size() == q.size()</tt>
 */
template<typename T, typename spvector_type> TProbSp<T,spvector_type> min( const TProbSp<T,spvector_type> &a, const TProbSp<T,spvector_type> &b ) {
    return a.pwBinaryTr( b, fo_min<T>() );
}


/// Returns the pointwise maximum of \a a and \a b
/** \relates TProbSp
 *  \pre <tt>this->size() == q.size()</tt>
 */
template<typename T, typename spvector_type> TProbSp<T,spvector_type> max( const TProbSp<T,spvector_type> &a, const TProbSp<T,spvector_type> &b ) {
    return a.pwBinaryTr( b, fo_max<T>() );
}


} // end of namespace dai


#endif
