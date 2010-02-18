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
/// \brief Defines TProbSpV<> class which represents sparse (probability) vectors (implemented using a vector)


#ifndef __defined_libdai_probspv_h
#define __defined_libdai_probspv_h


#include <cmath>
#include <vector>
#include <ostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <typeinfo>
#include <dai/util.h>
#include <dai/exceptions.h>
#include <dai/spvector.h>
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
template <typename T>
class TProbSpV {
    public:
        /// Type used to represent the sparse vector
        typedef spvector<T> spvector_type;

    private:
        /// The sparse vector containing nondefault elements
        spvector_type _p;

    public:
        /// Enumerates different ways of normalizing a probability measure.
        /**
         *  - NORMPROB means that the sum of all entries should be 1;
         *  - NORMLINF means that the maximum absolute value of all entries should be 1.
         */
        typedef enum { NORMPROB, NORMLINF } NormType;
        /// Enumerates different distance measures between probability measures.
        /**
         *  - DISTL1 is the \f$\ell_1\f$ distance (sum of absolute values of pointwise difference);
         *  - DISTLINF is the \f$\ell_\infty\f$ distance (maximum absolute value of pointwise difference);
         *  - DISTTV is the total variation distance (half of the \f$\ell_1\f$ distance);
         *  - DISTKL is the Kullback-Leibler distance (\f$\sum_i p_i (\log p_i - \log q_i)\f$).
         *  - DISTHEL is the Hellinger distance (\f$\frac{1}{2}\sum_i (\sqrt{p_i}-\sqrt{q_i})^2\f$).
         */
        typedef enum { DISTL1, DISTLINF, DISTTV, DISTKL, DISTHEL } DistType;

    /// \name Constructors and destructors
    //@{
        /// Default constructor (constructs empty vector)
        TProbSpV() : _p() {}

        /// Construct uniform probability distribution over \a n outcomes (i.e., a vector of length \a n with each entry set to \f$1/n\f$)
        explicit TProbSpV( size_t n ) : _p( n, (T)1 / n ) {}

        /// Construct vector of length \a n with each entry set to \a p
        explicit TProbSpV( size_t n, T p ) : _p( n, p ) {}

        /// Construct sparse vector from a range
        /** \tparam TIterator Iterates over instances that can be cast to \a T
         *  \param begin Points to first instance to be added.
         *  \param end Points just beyond last instance to be added.
         *  \param def Default value to use for the constructed sparse vector.
         *  \param sizeHint For efficiency, the number of nondefault entries can be speficied by \a sizeHint.
         */
        template <typename TIterator>
        TProbSpV( TIterator begin, TIterator end, T def=0, size_t sizeHint=0 ) : _p( begin, end, def, sizeHint ) {}

        /// Construct vector from another vector
        /** \tparam S type of elements in \a v (should be castable to type \a T)
         *  \param v vector used for initialization.
         *  \param def Default value to used for the constructed sparse vector.
         *  \param sizeHint For efficiency, the number of nondefault entries can be speficied by \a sizeHint.
         */
        template <typename S>
        TProbSpV( const std::vector<S> &v, T def=0, size_t sizeHint=0 ) : _p( v, def, sizeHint ) {}
    //@}

        /// Constant iterator over the elements
        typedef typename spvector_type::const_iterator const_iterator;
        /// Iterator over the elements
        typedef typename spvector_type::iterator iterator;
        /// Constant reverse iterator over the elements
        typedef typename spvector_type::const_reverse_iterator const_reverse_iterator;
        /// Reverse iterator over the elements
        typedef typename spvector_type::reverse_iterator reverse_iterator;

    /// @name Iterator interface
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

        /// Gets \a i 'th entry
        T get( size_t i ) const { return _p[i]; }

        /// Sets \a i 'th entry to \a val
        void set( size_t i, T val ) { _p.set( i, val ); }

    /// \name Queries
    //@{
        /// Returns a const reference to the wrapped map
        const spvector_type & p() const { return _p; }

        /// Returns a reference to the wrapped map
        spvector_type & p() { return _p; }

        /// Returns a copy of the \a i 'th entry
        T operator[]( size_t i ) const { return _p[i]; }

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

        /// Accumulate over all values, similar to std::accumulate
        template<typename binOp, typename unOp> T accumulate( T init, binOp op1, unOp op2 ) const {
            T t = init;
            for( const_iterator it = begin(); it != end(); it++ )
                t = op1( t, op2(it->second) );
            if( typeid(op1) == typeid(std::plus<T>()) )
                t += nrDef() * op2(def());
            else
                if( nrDef() )
                    t = op1( t, op2(def()) );
            return t;
        }

        /// Returns the Shannon entropy of \c *this, \f$-\sum_i p_i \log p_i\f$
        T entropy() const { return -accumulate( (T)0, std::plus<T>(), fo_plog0p<T>() ); }

        /// Returns maximum value of all entries
        T max() const { return accumulate( (T)(-INFINITY), fo_max<T>(), fo_id<T>() ); }

        /// Returns minimum value of all entries
        T min() const { return accumulate( (T)INFINITY, fo_min<T>(), fo_id<T>() ); }

        /// Returns sum of all entries
        T sum() const { return accumulate( (T)0, std::plus<T>(), fo_id<T>() ); }

        /// Return sum of absolute value of all entries
        T sumAbs() const { return accumulate( (T)0, std::plus<T>(), fo_abs<T>() ); }

        /// Returns maximum absolute value of all entries
        T maxAbs() const { return accumulate( (T)0, fo_max<T>(), fo_abs<T>() ); }

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
        bool operator<= (const TProbSpV<T> & q) const {
            DAI_DEBASSERT( size() == q.size() );
            for( size_t i = 0; i < size(); i++ )
                if( !(get(i) <= q.get(i)) )
                    return false;
            return true;
        }
    //@}

    /// \name Unary transformations
    //@{
        /// Returns the result of applying operation \a op pointwise on \c *this
        template<typename unaryOp> TProbSpV<T> pwUnaryTr( unaryOp op ) const {
            TProbSpV<T> r;
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
        TProbSp<T> operator- () const { return pwUnaryTr( std::negate<T>() ); }

        /// Returns pointwise absolute value
        TProbSpV<T> abs() const { return pwUnaryTr( fo_abs<T>() ); }

        /// Returns pointwise exponent
        TProbSpV<T> exp() const { return pwUnaryTr( fo_exp<T>() ); }

        /// Returns pointwise logarithm
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        TProbSpV<T> log(bool zero=false) const {
            if( zero )
                return pwUnaryTr( fo_log0<T>() );
            else
                return pwUnaryTr( fo_log<T>() );
        }

        /// Returns pointwise inverse
        /** If \a zero == \c true, uses <tt>1/0==0</tt>; otherwise, <tt>1/0==Inf</tt>.
         */
        TProbSpV<T> inverse(bool zero=true) const {
            if( zero )
                return pwUnaryTr( fo_inv0<T>() );
            else
                return pwUnaryTr( fo_inv<T>() );
        }

        /// Returns normalized copy of \c *this, using the specified norm
        /** \throw NOT_NORMALIZABLE if the norm is zero
         */
        TProbSpV<T> normalized( NormType norm = NORMPROB ) const {
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
        template<typename unaryOp> TProbSpV<T>& pwUnaryOp( unaryOp op ) {
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
        TProbSpV<T>& randomize() {
            setDef( 0 );
            for( size_t i = 0; i < size(); i++ )
                set( i, (T)rnd_uniform() );
            return *this;
        }

        /// Sets all entries to \f$1/n\f$ where \a n is the length of the vector
        TProbSpV<T>& setUniform () {
            setDef( (T)1 / size() );
            _p.clearNonDef();
            return *this;
        }

        /// Applies absolute value pointwise
        const TProbSpV<T>& takeAbs() { return pwUnaryOp( fo_abs<T>() ); }

        /// Applies exponent pointwise
        const TProbSpV<T>& takeExp() { return pwUnaryOp( fo_exp<T>() ); }

        /// Applies logarithm pointwise
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        const TProbSpV<T>& takeLog(bool zero=false) {
            if( zero ) {
                return pwUnaryOp( fo_log0<T>() );
            } else
                return pwUnaryOp( fo_log<T>() );
        }

        /// Normalizes vector using the specified norm
        /** \throw NOT_NORMALIZABLE if the norm is zero
         */
        T normalize( NormType norm=NORMPROB ) {
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
        TProbSpV<T> & fill( T x ) {
            setDef( x );
            _p.clearNonDef();
            return *this;
        }

        /// Adds scalar \a x to each entry
        TProbSpV<T>& operator+= (T x) {
            if( x != 0 )
                return pwUnaryOp( std::bind2nd( std::plus<T>(), x ) );
            else
                return *this;
        }

        /// Subtracts scalar \a x from each entry
        TProbSpV<T>& operator-= (T x) {
            if( x != 0 )
                return pwUnaryOp( std::bind2nd( std::minus<T>(), x ) );
            else
                return *this;
        }

        /// Multiplies each entry with scalar \a x
        TProbSpV<T>& operator*= (T x) {
            if( x != 1 )
                return pwUnaryOp( std::bind2nd( std::multiplies<T>(), x ) );
            else
                return *this;
        }

        /// Divides each entry by scalar \a x
        TProbSpV<T>& operator/= (T x) {
            DAI_DEBASSERT( x != 0 );
            if( x != 1 )
                return pwUnaryOp( std::bind2nd( std::divides<T>(), x ) );
            else
                return *this;
        }

        /// Raises entries to the power \a x
        TProbSpV<T>& operator^= (T x) {
            if( x != (T)1 )
                return pwUnaryOp( std::bind2nd( fo_pow<T>(), x) );
            else
                return *this;
        }
    //@}

    /// \name Transformations with scalars
    //@{
        /// Returns sum of \c *this and scalar \a x
        TProbSpV<T> operator+ (T x) const { return pwUnaryTr( std::bind2nd( std::plus<T>(), x ) ); }

        /// Returns difference of \c *this and scalar \a x
        TProbSpV<T> operator- (T x) const { return pwUnaryTr( std::bind2nd( std::minus<T>(), x ) ); }

        /// Returns product of \c *this with scalar \a x
        TProbSpV<T> operator* (T x) const { return pwUnaryTr( std::bind2nd( std::multiplies<T>(), x ) ); }

        /// Returns quotient of \c *this and scalar \a x, where division by 0 yields 0
        TProbSpV<T> operator/ (T x) const { return pwUnaryTr( std::bind2nd( fo_divides0<T>(), x ) ); }

        /// Returns \c *this raised to the power \a x
        TProbSpV<T> operator^ (T x) const { return pwUnaryTr( std::bind2nd( fo_pow<T>(), x ) ); }
    //@}

    /// \name Operations with other equally-sized vectors
    //@{
        /// Applies binary operation pointwise on two vectors
        /** \tparam binaryOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param q Right operand
         *  \param op Operation of type \a binaryOp
         */
        template<typename binaryOp> TProbSpV<T>& pwBinaryOp( const TProbSpV<T> &q, binaryOp op ) {
            DAI_DEBASSERT( size() == q.size() );
            TProbSpV<T> p(*this);
            setDef( op( p.def(), q.def() ) );
            for( typename TProbSpV<T>::const_iterator it = p.begin(); it != p.end(); it++ ) {
                T new_val = op( it->second, q[it->first] );
                set( it->first, new_val );
            }
            for( typename TProbSpV<T>::const_iterator it = q.begin(); it != q.end(); it++ ) {
                T new_val = op( p[it->first], it->second );
                set( it->first, new_val );
            }
            return *this;
        }

        /// Pointwise addition with \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T>& operator+= (const TProbSpV<T> & q) { return pwBinaryOp( q, std::plus<T>() ); }

        /// Pointwise subtraction of \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T>& operator-= (const TProbSpV<T> & q) { return pwBinaryOp( q, std::minus<T>() ); }

        /// Pointwise multiplication with \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T>& operator*= (const TProbSpV<T> & q) { return pwBinaryOp( q, std::multiplies<T>() ); }

        /// Pointwise division by \a q, where division by 0 yields 0
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see divide(const TProbSpV<T> &)
         */
        TProbSpV<T>& operator/= (const TProbSpV<T> & q) { return pwBinaryOp( q, fo_divides0<T>() ); }

        /// Pointwise division by \a q, where division by 0 yields +Inf
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see operator/=(const TProbSpV<T> &)
         */
        TProbSpV<T>& divide (const TProbSpV<T> & q) { return pwBinaryOp( q, std::divides<T>() ); }

        /// Pointwise power
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T>& operator^= (const TProbSpV<T> & q) { return pwBinaryOp( q, fo_pow<T>() ); }
    //@}

    /// \name Transformations with other equally-sized vectors
    //@{
        /// Returns the result of applying binary operation \a op pointwise on \c *this and \a q
        /** \tparam binaryOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param q Right operand
         *  \param op Operation of type \a binaryOp
         */
        template<typename binaryOp> TProbSpV<T> pwBinaryTr( const TProbSpV<T> &q, binaryOp op ) const {
            DAI_DEBASSERT( size() == q.size() );
            TProbSpV<T> result;
            result.setDef( op( def(), q.def() ) );
            result._p.resize( size() );
            for( typename TProbSpV<T>::const_iterator it = begin(); it != end(); it++ ) {
                T new_val = op( it->second, q[it->first] );
                if( new_val != result.def() )
                    result._p.push_back(it->first, new_val);
            }
            for( typename TProbSpV<T>::const_iterator it = q.begin(); it != q.end(); it++ ) {
                T new_val = op( get(it->first), it->second );
                if( new_val != result.def() )
                    result._p.push_back(it->first, new_val);
            }
            return result;
        }

        /// Returns sum of \c *this and \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T> operator+ ( const TProbSpV<T>& q ) const { return pwBinaryTr( q, std::plus<T>() ); }

        /// Return \c *this minus \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T> operator- ( const TProbSpV<T>& q ) const { return pwBinaryTr( q, std::minus<T>() ); }

        /// Return product of \c *this with \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T> operator* ( const TProbSpV<T> &q ) const { return pwBinaryTr( q, std::multiplies<T>() ); }

        /// Returns quotient of \c *this with \a q, where division by 0 yields 0
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see divided_by(const TProbSpV<T> &)
         */
        TProbSpV<T> operator/ ( const TProbSpV<T> &q ) const { return pwBinaryTr( q, fo_divides0<T>() ); }

        /// Pointwise division by \a q, where division by 0 yields +Inf
        /** \pre <tt>this->size() == q.size()</tt>
         *  \see operator/(const TProbSpV<T> &)
         */
        TProbSpV<T> divided_by( const TProbSpV<T> &q ) const { return pwBinaryTr( q, std::divides<T>() ); }

        /// Returns \c *this to the power \a q
        /** \pre <tt>this->size() == q.size()</tt>
         */
        TProbSpV<T> operator^ ( const TProbSpV<T> &q ) const { return pwBinaryTr( q, fo_pow<T>() ); }
    //@}

        /// Performs a generalized inner product, similar to std::inner_product
        /** \pre <tt>this->size() == q.size()</tt>
         */
        template<typename binOp1, typename binOp2> T innerProduct( const TProbSpV<T> &q, T init, binOp1 binaryOp1, binOp2 binaryOp2 ) const {
            DAI_DEBASSERT( size() == q.size() );
            DAI_THROW(NOT_IMPLEMENTED);
            return std::inner_product( begin(), end(), q.begin(), init, binaryOp1, binaryOp2 );
        }
};


/// Returns distance between \a p and \a q, measured using distance measure \a dt
/** \relates TProbSpV
 *  \pre <tt>this->size() == q.size()</tt>
 */
template<typename T> T dist( const TProbSpV<T> &p, const TProbSpV<T> &q, typename TProbSpV<T>::DistType dt ) {
    switch( dt ) {
        case TProbSpV<T>::DISTL1:
            return (p - q).sumAbs();
        case TProbSpV<T>::DISTLINF:
            return (p - q).maxAbs();
        case TProbSpV<T>::DISTTV:
            return (p - q).sumAbs() / 2;
        case TProbSpV<T>::DISTKL:
            return p.pwBinaryTr( q, fo_KL<T>() ).sum();
        case TProbSpV<T>::DISTHEL:
            return p.pwBinaryTr( q, fo_Hellinger<T>() ).sum() / 2;
        default:
            DAI_THROW(UNKNOWN_ENUM_VALUE);
            return INFINITY;
    }
}


/// Writes a TProbSp<T> to an output stream
/** \relates TProbSpV
 */
template<typename T> std::ostream& operator<< (std::ostream& os, const TProbSpV<T>& p) {
    os << p.p();
    return os;
}


/// Returns the pointwise minimum of \a a and \a b
/** \relates TProbSpV
 *  \pre <tt>this->size() == q.size()</tt>
 */
template<typename T> TProbSpV<T> min( const TProbSpV<T> &a, const TProbSpV<T> &b ) {
    return a.pwBinaryTr( b, fo_min<T>() );
}


/// Returns the pointwise maximum of \a a and \a b
/** \relates TProbSpV
 *  \pre <tt>this->size() == q.size()</tt>
 */
template<typename T> TProbSpV<T> max( const TProbSpV<T> &a, const TProbSpV<T> &b ) {
    return a.pwBinaryTr( b, fo_max<T>() );
}


} // end of namespace dai


#endif
