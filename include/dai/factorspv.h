/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  libDAI is licensed under the terms of the GNU General Public License version
 *  2, or (at your option) any later version. libDAI is distributed without any
 *  warranty. See the file COPYING for more details.
 *
 *  Copyright (C) 2002       Martijn Leisink  [martijn@mbfys.kun.nl]
 *  Copyright (C) 2006-2009  Joris Mooij      [joris dot mooij at libdai dot org]
 *  Copyright (C) 2002-2007  Radboud University Nijmegen, The Netherlands
 */


/// \file
/// \brief Defines TFactorSpV<> class which represents sparse factors in probability distributions.


#ifndef __defined_libdai_factorspv_h
#define __defined_libdai_factorspv_h


#include <iostream>
#include <functional>
#include <cmath>
#include <dai/prob.h>
#include <dai/varset.h>
#include <dai/index.h>
#include <dai/util.h>


namespace dai {


/// Represents a (probability) factor.
/** Mathematically, a \e factor is a function mapping joint states of some
 *  variables to the nonnegative real numbers.
 *  More formally, denoting a discrete variable with label \f$l\f$ by
 *  \f$x_l\f$ and its state space by \f$X_l = \{0,1,\dots,S_l-1\}\f$,
 *  a factor depending on the variables \f$\{x_l\}_{l\in L}\f$ is
 *  a function \f$f_L : \prod_{l\in L} X_l \to [0,\infty)\f$.
 *
 *  In libDAI, a factor is represented by a TFactorSpV<T> object, which has two
 *  components:
 *  \arg a VarSet, corresponding with the set of variables \f$\{x_l\}_{l\in L}\f$
 *  that the factor depends on;
 *  \arg a TProbSpV, a vector containing the value of the factor for each possible
 *  joint state of the variables.
 *
 *  The factor values are stored in the entries of the TProbSpV in a particular
 *  ordering, which is defined by the one-to-one correspondence of a joint state
 *  in \f$\prod_{l\in L} X_l\f$ with a linear index in
 *  \f$\{0,1,\dots,\prod_{l\in L} S_l-1\}\f$ according to the mapping \f$\sigma\f$
 *  induced by VarSet::calcState(const std::map<Var,size_t> &).
 *
 *  \tparam T Should be a scalar that is castable from and to double and should support elementary arithmetic operations.
 *  \todo Define a better fileformat for .fg files (maybe using XML)?
 *  \todo Add support for sparse factors.
 */
template <typename T> class TFactorSpV {
    private:
        /// Stores the variables on which the factor depends
        VarSet      _vs;
        /// Stores the factor values
        TProbSpV<T>    _p;

    public:
    /// \name Constructors and destructors
    //@{
        /// Constructs factor depending on no variables with value \a p
        TFactorSpV ( T p = 1 ) : _vs(), _p(1,p) {}

        /// Constructs factor depending on the variable \a v with uniform distribution
        TFactorSpV( const Var &v ) : _vs(v), _p(v.states()) {}

        /// Constructs factor depending on variables in \a vars with uniform distribution
        TFactorSpV( const VarSet& vars ) : _vs(vars), _p(_vs.nrStates()) {}

        /// Constructs factor depending on variables in \a vars with all values set to \a p
        TFactorSpV( const VarSet& vars, T p ) : _vs(vars), _p(_vs.nrStates(),p) {}

        /// Constructs factor depending on variables in \a vars, copying the values from a std::vector<>
        /** \tparam S Type of values of \a x
         *  \param vars contains the variables that the new factor should depend on.
         *  \param x Vector with values to be copied.
         */
        template<typename S>
        TFactorSpV( const VarSet& vars, const std::vector<S> &x ) : _vs(vars), _p(x.begin(), x.begin() + _vs.nrStates(), _vs.nrStates()) {
            DAI_ASSERT( x.size() == vars.nrStates() );
        }

        /// Constructs factor depending on variables in \a vars, copying the values from an array
        /** \param vars contains the variables that the new factor should depend on.
         *  \param p Points to array of values to be added.
         */
        TFactorSpV( const VarSet& vars, const T* p ) : _vs(vars), _p(p, p + _vs.nrStates(), _vs.nrStates()) {}

        /// Constructs factor depending on variables in \a vars, copying the values from \a p
        TFactorSpV( const VarSet& vars, const TProbSpV<T> &p ) : _vs(vars), _p(p) {
            DAI_DEBASSERT( _vs.nrStates() == _p.size() );
        }

        /// Constructs factor depending on variables in \a vars, permuting the values given in \a p accordingly
        TFactorSpV( const std::vector<Var> &vars, const std::vector<T> &p ) : _vs(vars.begin(), vars.end(), vars.size()), _p(p.size()) {
            Permute permindex(vars);
            for( size_t li = 0; li < p.size(); ++li )
                _p.set( permindex.convertLinearIndex(li), p[li] );
        }
    //@}

    /// \name Get/set individual entries
    //@{
        /// Sets \a i 'th entry to \a val
        void set( size_t i, T val ) { _p.set( i, val ); }

        /// Gets \a i 'th entry
        T get( size_t i ) const { return _p[i]; }
    //@}

    /// \name Queries
    //@{
        /// Returns constant reference to value vector
        const TProbSpV<T>& p() const { return _p; }

        /// Returns reference to value vector
        TProbSpV<T>& p() { return _p; }

        /// Returns a copy of the \a i 'th entry of the value vector
        T operator[] (size_t i) const { return _p[i]; }

        /// Returns constant reference to variable set (i.e., the variables on which the factor depends)
        const VarSet& vars() const { return _vs; }

        /// Returns reference to variable set (i.e., the variables on which the factor depends)
        VarSet& vars() { return _vs; }

        /// Returns the number of possible joint states of the variables on which the factor depends, \f$\prod_{l\in L} S_l\f$
        /** \note This is equal to the length of the value vector.
         */
        size_t states() const { return _p.size(); }

        /// Returns the Shannon entropy of \c *this, \f$-\sum_i p_i \log p_i\f$
        T entropy() const { return _p.entropy(); }

        /// Returns maximum of all values
        T max() const { return _p.max(); }

        /// Returns minimum of all values
        T min() const { return _p.min(); }

        /// Returns sum of all values
        T sum() const { return _p.sum(); }

        /// Returns maximum absolute value of all values
        T maxAbs() const { return _p.maxAbs(); }

        /// Returns \c true if one or more values are NaN
        bool hasNaNs() const { return _p.hasNaNs(); }

        /// Returns \c true if one or more values are negative
        bool hasNegatives() const { return _p.hasNegatives(); }

        /// Returns strength of this factor (between variables \a i and \a j), as defined in eq. (52) of [\ref MoK07b]
        T strength( const Var &i, const Var &j ) const;
    //@}

    /// \name Unary transformations
    //@{
        /// Returns pointwise absolute value
        TFactorSpV<T> abs() const {
            // Note: the alternative (shorter) way of implementing this,
            //   return TFactorSpV<T>( _vs, _p.abs() );
            // is slower because it invokes the copy constructor of TProbSpV<T>
            TFactorSpV<T> x;
            x._vs = _vs;
            x._p = _p.abs();
            return x;
        }

        /// Returns pointwise exponent
        TFactorSpV<T> exp() const {
            TFactorSpV<T> x;
            x._vs = _vs;
            x._p = _p.exp();
            return x;
        }

        /// Returns pointwise logarithm
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        TFactorSpV<T> log(bool zero=false) const {
            TFactorSpV<T> x;
            x._vs = _vs;
            x._p = _p.log(zero);
            return x;
        }

        /// Returns pointwise inverse
        /** If \a zero == \c true, uses <tt>1/0==0</tt>; otherwise, <tt>1/0==Inf</tt>.
         */
        TFactorSpV<T> inverse(bool zero=true) const {
            TFactorSpV<T> x;
            x._vs = _vs;
            x._p = _p.inverse(zero);
            return x;
        }

        /// Returns normalized copy of \c *this, using the specified norm
        TFactorSpV<T> normalized( typename TProbSpV<T>::NormType norm=TProbSpV<T>::NORMPROB ) const {
            TFactorSpV<T> x;
            x._vs = _vs;
            x._p = _p.normalized( norm );
            return x;
        }
    //@}

    /// \name Unary operations
    //@{
        /// Draws all values i.i.d. from a uniform distribution on [0,1)
        TFactorSpV<T> & randomize () { _p.randomize(); return *this; }

        /// Sets all values to \f$1/n\f$ where \a n is the number of states
        TFactorSpV<T>& setUniform () { _p.setUniform(); return *this; }

        /// Normalizes factor using the specified norm
        T normalize( typename TProbSpV<T>::NormType norm=TProbSpV<T>::NORMPROB ) { return _p.normalize( norm ); }
    //@}

    /// \name Operations with scalars
    //@{
        /// Sets all values to \a x
        TFactorSpV<T> & fill (T x) { _p.fill( x ); return *this; }

        /// Adds scalar \a x to each value
        TFactorSpV<T>& operator+= (T x) { _p += x; return *this; }

        /// Subtracts scalar \a x from each value
        TFactorSpV<T>& operator-= (T x) { _p -= x; return *this; }

        /// Multiplies each value with scalar \a x
        TFactorSpV<T>& operator*= (T x) { _p *= x; return *this; }

        /// Divides each entry by scalar \a x
        TFactorSpV<T>& operator/= (T x) { _p /= x; return *this; }

        /// Raises values to the power \a x
        TFactorSpV<T>& operator^= (T x) { _p ^= x; return *this; }
    //@}

    /// \name Transformations with scalars
    //@{
        /// Returns sum of \c *this and scalar \a x
        TFactorSpV<T> operator+ (T x) const {
            // Note: the alternative (shorter) way of implementing this,
            //   TFactorSpV<T> result(*this);
            //   result._p += x;
            // is slower because it invokes the copy constructor of TFactorSpV<T>
            TFactorSpV<T> result;
            result._vs = _vs;
            result._p = p() + x;
            return result;
        }

        /// Returns difference of \c *this and scalar \a x
        TFactorSpV<T> operator- (T x) const {
            TFactorSpV<T> result;
            result._vs = _vs;
            result._p = p() - x;
            return result;
        }

        /// Returns product of \c *this with scalar \a x
        TFactorSpV<T> operator* (T x) const {
            TFactorSpV<T> result;
            result._vs = _vs;
            result._p = p() * x;
            return result;
        }

        /// Returns quotient of \c *this with scalar \a x
        TFactorSpV<T> operator/ (T x) const {
            TFactorSpV<T> result;
            result._vs = _vs;
            result._p = p() / x;
            return result;
        }

        /// Returns \c *this raised to the power \a x
        TFactorSpV<T> operator^ (T x) const {
            TFactorSpV<T> result;
            result._vs = _vs;
            result._p = p() ^ x;
            return result;
        }
    //@}

    /// \name Operations with other factors
    //@{
        /// Applies binary operation \a op on two factors, \c *this and \a g
        /** \tparam binOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param g Right operand
         *  \param op Operation of type \a binOp
         */
        template<typename binOp> TFactorSpV<T>& binaryOp( const TFactorSpV<T> &g, binOp op ) {
            if( _vs == g._vs ) // optimize special case
                _p.pwBinaryOp( g._p, op );
            else {
                *this = pointwiseOp( *this, g, op );
                // OPTIMIZE ME
/*                TFactorSpV<T> f(*this); // make a copy
                _vs |= g._vs;
                size_t N = _vs.nrStates();

                IndexFor i_f( f._vs, _vs );
                IndexFor i_g( g._vs, _vs );

                _p.p().clear();
                _p.p().reserve( N );
                for( size_t i = 0; i < N; i++, ++i_f, ++i_g )
                    _p.p().push_back( op( f[i_f], g[i_g] ) );*/
            }
            return *this;
        }

        /// Adds \a g to \c *this
        /** The sum of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f+g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) + g(x_M).\f]
         */
        TFactorSpV<T>& operator+= (const TFactorSpV<T>& g) { return binaryOp( g, std::plus<T>() ); }

        /// Subtracts \a g from \c *this
        /** The difference of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f-g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) - g(x_M).\f]
         */
        TFactorSpV<T>& operator-= (const TFactorSpV<T>& g) { return binaryOp( g, std::minus<T>() ); }

        /// Multiplies \c *this with \a g
        /** The product of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[fg : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) g(x_M).\f]
         */
        TFactorSpV<T>& operator*= (const TFactorSpV<T>& g) {
            // Note that the following implementation is slow, because it doesn't exploit the special case of value 0
            //   return binaryOp( g, std::multiplies<T>() );
            if( _vs == g._vs ) // optimize special case
                _p.pwBinaryOp( g._p, std::multiplies<T>() );
            else
                *this = pointwiseOp( *this, g, std::multiplies<T>(), p().def() == (T)0 && g.p().def() == (T)0 );
            return *this;
        }

        /// Divides \c *this by \a g (where division by zero yields zero)
        /** The quotient of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[\frac{f}{g} : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto \frac{f(x_L)}{g(x_M)}.\f]
         */
        TFactorSpV<T>& operator/= (const TFactorSpV<T>& g) { return binaryOp( g, fo_divides0<T>() ); }
    //@}

    /// \name Transformations with other factors
    //@{
        /// Returns result of applying binary operation \a op on two factors, \c *this and \a g
        /** \tparam binOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param g Right operand
         *  \param op Operation of type \a binOp
         */
        template<typename binOp> TFactorSpV<T> binaryTr( const TFactorSpV<T> &g, binOp op ) const {
            // OPTIMIZE ME
            return pointwiseOp( *this, g, op );
            // Note that to prevent a copy to be made, it is crucial 
            // that the result is declared outside the if-else construct.
/*            TFactorSpV<T> result;
            if( _vs == g._vs ) { // optimize special case
                result._vs = _vs;
                result._p = _p.pwBinaryTr( g._p, op );
            } else {
                result._vs = _vs | g._vs;
                size_t N = result._vs.nrStates();

                IndexFor i_f( _vs, result.vars() );
                IndexFor i_g( g._vs, result.vars() );

                result._p.p().clear();
                result._p.p().reserve( N );
                for( size_t i = 0; i < N; i++, ++i_f, ++i_g )
                    result._p.p().push_back( op( _p[i_f], g[i_g] ) );
            }
            return result;*/
        }

        /// Returns sum of \c *this and \a g
        /** The sum of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f+g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) + g(x_M).\f]
         */
        TFactorSpV<T> operator+ (const TFactorSpV<T>& g) const {
            return binaryTr(g,std::plus<T>());
        }

        /// Returns \c *this minus \a g
        /** The difference of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f-g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) - g(x_M).\f]
         */
        TFactorSpV<T> operator- (const TFactorSpV<T>& g) const {
            return binaryTr(g,std::minus<T>());
        }

        /// Returns product of \c *this with \a g
        /** The product of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[fg : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) g(x_M).\f]
         */
        TFactorSpV<T> operator* (const TFactorSpV<T>& g) const {
            return pointwiseOp( *this, g, std::multiplies<T>(), p().def() == (T)0 && g.p().def() == (T)0 );
        }

        /// Returns quotient of \c *this by \a f (where division by zero yields zero)
        /** The quotient of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[\frac{f}{g} : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto \frac{f(x_L)}{g(x_M)}.\f]
         */
        TFactorSpV<T> operator/ (const TFactorSpV<T>& g) const {
            return binaryTr(g,fo_divides0<T>());
        }
    //@}

    /// \name Miscellaneous operations
    //@{
        /// Returns a slice of \c *this, where the subset \a vars is in state \a varsState
        /** \pre \a vars sould be a subset of vars()
         *  \pre \a varsState < vars.states()
         *
         *  The result is a factor that depends on the variables of *this except those in \a vars,
         *  obtained by setting the variables in \a vars to the joint state specified by the linear index
         *  \a varsState. Formally, if \c *this corresponds with the factor \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$,
         *  \f$M \subset L\f$ corresponds with \a vars and \a varsState corresponds with a mapping \f$s\f$ that
         *  maps a variable \f$x_m\f$ with \f$m\in M\f$ to its state \f$s(x_m) \in X_m\f$, then the slice
         *  returned corresponds with the factor \f$g : \prod_{l \in L \setminus M} X_l \to [0,\infty)\f$
         *  defined by \f$g(\{x_l\}_{l\in L \setminus M}) = f(\{x_l\}_{l\in L \setminus M}, \{s(x_m)\}_{m\in M})\f$.
         */
        TFactorSpV<T> slice( const VarSet& vars, size_t varsState ) const;

        /// Embeds this factor in a larger VarSet
        /** \pre vars() should be a subset of \a vars 
         *
         *  If *this corresponds with \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$L \subset M\f$, then
         *  the embedded factor corresponds with \f$g : \prod_{m\in M} X_m \to [0,\infty) : x \mapsto f(x_L)\f$.
         */
        TFactorSpV<T> embed(const VarSet & vars) const {
            DAI_ASSERT( vars >> _vs );
            if( _vs == vars )
                return *this;
            else
                return (*this) * TFactorSpV<T>(vars / _vs, (T)1);
        }

        /// Returns marginal on \a vars, obtained by summing out all variables except those in \a vars, and normalizing the result if \a normed == \c true
        TFactorSpV<T> marginal(const VarSet &vars, bool normed=true) const;

        /// Returns max-marginal on \a vars, obtained by maximizing all variables except those in \a vars, and normalizing the result if \a normed == \c true
        TFactorSpV<T> maxMarginal(const VarSet &vars, bool normed=true) const;
    //@}
};


template<typename T> TFactorSpV<T> TFactorSpV<T>::slice( const VarSet& vars, size_t varsState ) const {
    DAI_ASSERT( vars << _vs );
    VarSet varsrem = _vs / vars;

    TFactorSpV<T> result( varsrem, p().def() );
    for( typename TProbSpV<T>::const_iterator it = p().begin(); it != p().end(); it++ ) {
        State s( _vs, it->first );
        size_t vars_s = s( vars );
        if( vars_s == varsState )
            result.set( s(varsrem), it->second );
    }

    /* SLOW BECAUSE IT ITERATES OVER ALL VALUES */
    // OPTIMIZE ME
/*  TFactorSpV<T> res( varsrem, T(0) );
    IndexFor i_vars (vars, _vs);
    IndexFor i_varsrem (varsrem, _vs);
    for( size_t i = 0; i < states(); i++, ++i_vars, ++i_varsrem )
        if( (size_t)i_vars == varsState )
            res.set( i_varsrem, _p[i] );

    if( !((result.p() <= res.p()) && (res.p() <= result.p())) ) {
        std::cerr << result << std::endl;
        std::cerr << res << std::endl;
        DAI_ASSERT( ((result.p() <= res.p()) && (res.p() <= result.p())) );
    }*/

    return result;
}


template<typename T> TFactorSpV<T> TFactorSpV<T>::marginal(const VarSet &vars, bool normed) const {
    VarSet res_vars = vars & _vs;

    VarSet rem(_vs / res_vars);
    TFactorSpV<T> result( res_vars, rem.nrStates() * p().def() );
    for( typename TProbSpV<T>::const_iterator it = p().begin(); it != p().end(); it++ ) {
        State s(_vs, it->first);
        size_t res_vars_s = s( res_vars );
        result.set( res_vars_s, result[res_vars_s] - p().def() + it->second );
    }

    /* SLOW BECAUSE IT ITERATES OVER ALL VALUES
    TFactorSpV<T> res( res_vars, 0.0 );
    IndexFor i_res( res_vars, _vs );
    for( size_t i = 0; i < _p.size(); i++, ++i_res )
        res.set( i_res, res[i_res] + _p[i] );

    if( !((result.p() <= res.p()) && (res.p() <= result.p())) ) {
        std::cerr << result << std::endl;
        std::cerr << res << std::endl;
        DAI_ASSERT( ((result.p() <= res.p()) && (res.p() <= result.p())) );
    }
    */

    if( normed )
        result.normalize( TProbSpV<T>::NORMPROB );

    return result;
}


template<typename T> TFactorSpV<T> TFactorSpV<T>::maxMarginal(const VarSet &vars, bool normed) const {
    VarSet res_vars = vars & _vs;

    VarSet rem(_vs / res_vars);
    TFactorSpV<T> result( res_vars, p().def() );
    for( typename TProbSpV<T>::const_iterator it = p().begin(); it != p().end(); it++ ) {
        State s( _vs, it->first );
        size_t res_vars_s = s( res_vars );
        if( it->second > result[res_vars_s] )
            result.set( res_vars_s, it->second );
    }

    /* SLOW BECAUSE IT ITERATES OVER ALL VALUES */
/*
    TFactorSpV<T> res( res_vars, 0.0 );
    IndexFor i_res( res_vars, _vs );
    for( size_t i = 0; i < _p.size(); i++, ++i_res )
        if( _p[i] > res._p[i_res] )
            res.set( i_res, _p[i] );

    if( !((result.p() <= res.p()) && (res.p() <= result.p())) ) {
        std::cerr << result << std::endl;
        std::cerr << res << std::endl;
        DAI_ASSERT( ((result.p() <= res.p()) && (res.p() <= result.p())) );
    }
*/
    if( normed )
        result.normalize( TProbSpV<T>::NORMPROB );

    return result;
}


template<typename T> T TFactorSpV<T>::strength( const Var &i, const Var &j ) const {
    DAI_DEBASSERT( _vs.contains( i ) );
    DAI_DEBASSERT( _vs.contains( j ) );
    DAI_DEBASSERT( i != j );
    VarSet ij(i, j);

    T max = 0.0;
    for( size_t alpha1 = 0; alpha1 < i.states(); alpha1++ )
        for( size_t alpha2 = 0; alpha2 < i.states(); alpha2++ )
            if( alpha2 != alpha1 )
                for( size_t beta1 = 0; beta1 < j.states(); beta1++ )
                    for( size_t beta2 = 0; beta2 < j.states(); beta2++ )
                        if( beta2 != beta1 ) {
                            size_t as = 1, bs = 1;
                            if( i < j )
                                bs = i.states();
                            else
                                as = j.states();
                            T f1 = slice( ij, alpha1 * as + beta1 * bs ).p().divide( slice( ij, alpha2 * as + beta1 * bs ).p() ).max();
                            T f2 = slice( ij, alpha2 * as + beta2 * bs ).p().divide( slice( ij, alpha1 * as + beta2 * bs ).p() ).max();
                            T f = f1 * f2;
                            if( f > max )
                                max = f;
                        }

    return std::tanh( 0.25 * std::log( max ) );
}


/// Apply binary operator pointwise on two factors
/** \relates TFactorSpV
 *  \tparam binaryOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
 *  \param f Left operand
 *  \param g Right operand
 *  \param op Operation of type \a binaryOp
 *  \param fast If true, supposes that the default value of \a f always gives the same result in op, and
 *         similarly for the default value of \a g
 */
template<typename T, typename binaryOp> TFactorSpV<T> pointwiseOp( const TFactorSpV<T> &f, const TFactorSpV<T> &g, binaryOp op, bool fast=false ) {
    if( f.vars() == g.vars() ) { // optimizate special case
        TFactorSpV<T> result( f.vars() );
        result.p() = f.p().pwBinaryTr( g.p(), op );
        return result;
    } else {
        // Union of variables
        VarSet un( f.vars() | g.vars() );
        // Intersection of variables
        VarSet is( f.vars() & g.vars() );
        // Result factor
        TFactorSpV<T> result( un, op( f.p().def(), g.p().def() ) );

        if( fast ) {
            // For all non-default states of f and all non-default states of g
            for( typename TProbSpV<T>::const_iterator fit = f.p().begin(); fit != f.p().end(); fit++ ) {
                // calculate state of f
                State fs( f.vars(), fit->first );
                for( typename TProbSpV<T>::const_iterator git = g.p().begin(); git != g.p().end(); git++ ) {
                    // calculate state of g
                    State gs( g.vars(), git->first );
                    // check whether these states are compatible
                    bool compatible = true;
                    for( typename VarSet::const_iterator v = is.begin(); v != is.end() && compatible; v++ )
                        if( fs(*v) != gs(*v) )
                            compatible = false;
                    if( compatible ) {
                        State fgs = fs;
                        fgs.insert( gs.begin(), gs.end() );
                        result.set( fgs(un), op( fit->second, git->second ) );
                    }
                }
            }
        } else {
            // For all non-default states of f and all states of g
            for( typename TProbSpV<T>::const_iterator fit = f.p().begin(); fit != f.p().end(); fit++ ) {
                State fs( f.vars(), fit->first );
                for( State g_minus_f_s(g.vars() / f.vars()); g_minus_f_s.valid(); g_minus_f_s++ ) {
                    State fgs = g_minus_f_s.get();
                    fgs.insert( fs.begin(), fs.end() );
                    result.set( fgs(un), op( fit->second, g[fgs(g.vars())] ) );
                }
            }
            // For all states of f and all non-default states of g
            for( typename TProbSpV<T>::const_iterator git = g.p().begin(); git != g.p().end(); git++ ) {
                State gs( g.vars(), git->first );
                for( State f_minus_g_s(f.vars() / g.vars()); f_minus_g_s.valid(); f_minus_g_s++ ) {
                    State fgs = f_minus_g_s.get();
                    fgs.insert( gs.begin(), gs.end() );
                    result.set( fgs(un), op( f[fgs(f.vars())], git->second ) );
                }
            }
        }

        /* SLOW BECAUSE IT ITERATES OVER ALL VALUES */
/*        TFactorSpV<T> resultold( un, op( f.p().def(), g.p().def() ) );
        IndexFor i1(f.vars(), result.vars());
        IndexFor i2(g.vars(), result.vars());

        for( size_t i = 0; i < result.states(); i++, ++i1, ++i2 )
            resultold.set(i, op( f[i1], g[i2] ));

        if( !((result.p() <= resultold.p()) && (resultold.p() <= result.p())) ) {
            std::cerr << result << std::endl;
            std::cerr << resultold << std::endl;
            DAI_ASSERT( ((result.p() <= resultold.p()) && (resultold.p() <= result.p())) );
        }
*/

        return result;
    }
}


/// Writes a factor to an output stream
/** \relates TFactorSpV
 */
template<typename T> std::ostream& operator<< (std::ostream& os, const TFactorSpV<T>& f) {
//    os << "(" << f.vars() << ", " << f.p() << ")";
    os << "(" << f.vars() << ", (";
    for( size_t i = 0; i < f.states(); i++ )
        os << (i == 0 ? "" : ", ") << f[i];
    os << "))";
    return os;
}


/// Returns distance between two factors \a f and \a g, according to the distance measure \a dt
/** \relates TFactorSpV
 *  \pre f.vars() == g.vars()
 */
template<typename T> T dist( const TFactorSpV<T> &f, const TFactorSpV<T> &g, typename TProbSpV<T>::DistType dt ) {
    if( f.vars().empty() || g.vars().empty() )
        return -1;
    else {
        DAI_DEBASSERT( f.vars() == g.vars() );
        return dist( f.p(), g.p(), dt );
    }
}


/// Returns the pointwise maximum of two factors
/** \relates TFactorSpV
 *  \pre f.vars() == g.vars()
 */
template<typename T> TFactorSpV<T> max( const TFactorSpV<T> &f, const TFactorSpV<T> &g ) {
    DAI_ASSERT( f._vs == g._vs );
    return TFactorSpV<T>( f._vs, max( f.p(), g.p() ) );
}


/// Returns the pointwise minimum of two factors
/** \relates TFactorSpV
 *  \pre f.vars() == g.vars()
 */
template<typename T> TFactorSpV<T> min( const TFactorSpV<T> &f, const TFactorSpV<T> &g ) {
    DAI_ASSERT( f._vs == g._vs );
    return TFactorSpV<T>( f._vs, min( f.p(), g.p() ) );
}


/// Calculates the mutual information between the two variables that \a f depends on, under the distribution given by \a f
/** \relates TFactorSpV
 *  \pre f.vars().size() == 2
 */
template<typename T> T MutualInfo(const TFactorSpV<T> &f) {
    DAI_ASSERT( f.vars().size() == 2 );
    VarSet::const_iterator it = f.vars().begin();
    Var i = *it; it++; Var j = *it;
    TFactorSpV<T> projection = f.marginal(i) * f.marginal(j);
    return dist( f.normalized(), projection, TProbSpV<T>::DISTKL );
}


} // end of namespace dai


#endif
