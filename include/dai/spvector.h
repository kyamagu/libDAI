/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  libDAI is licensed under the terms of the GNU General Public License version
 *  2, or (at your option) any later version. libDAI is distributed without any
 *  warranty. See the file COPYING for more details.
 *
 *  Copyright (C) 2010  Joris Mooij  [joris dot mooij at libdai dot org]
 *  Copyright (C) 2010  Vicenç Gomez [v dot gomez at science dot ru dot nl]
 *  Copyright (C) 2010  Radboud University Nijmegen, The Netherlands
 */


/// \file
/// \brief Defines spvector<> class, which implements a sparse vector (using a std::vector)


#ifndef __defined_libdai_spvector_h
#define __defined_libdai_spvector_h


#include <ostream>
#include <vector>


namespace dai {


/// Function object that returns true if a.first < b.first
template <typename T1, typename T2>
struct first_less : public std::binary_function<const std::pair<T1,T2> &, const std::pair<T1,T2> &, bool> {
    bool operator()( const std::pair<T1,T2> &a, const std::pair<T1,T2> &b ) { return a.first < b.first; }
};


template <typename T>
class spvector {
    public:
        /// Index-value pair representing an entry of a sparse vector
        typedef std::pair<size_t, T> nondefault_type;

        /// Type of container for nondefault entries
        typedef std::vector<nondefault_type> nondefaults_type;

    private:
        /// The container containing all nondefault elements
        /** \note The entries are sorted ascendingly according to their index
         */
        nondefaults_type _p;
        /// Indices range from 0, 1, ..., _size - 1
        size_t _size;
        /// Default value
        T _def;

    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor (constructs empty vector)
        spvector() : _p(), _size(0), _def(T()) {}
        
        /// Construct vector of length \a n with each entry set to \a p
        explicit spvector( size_t n, T p ) : _p(), _size(n), _def(p) {}

        /// Construct sparse vector from a range
        /** \tparam TIterator Iterates over instances that can be cast to \a T
         *  \param begin Points to first instance to be added.
         *  \param end Points just beyond last instance to be added.
         *  \param def Default value to use for the constructed sparse vector.
         *  \param sizeHint For efficiency, the number of nondefault entries can be speficied by \a sizeHint.
         */
        template <typename TIterator>
        spvector( TIterator begin, TIterator end, T def=T(), size_t sizeHint=0 ) : _p(), _size(0), _def(def) {
            if( sizeHint )
                reserve( sizeHint );
            size_t iter = 0;
            for( TIterator it = begin; it != end; it++, iter++ )
                if( *it != def )
                    push_back( iter, *it );
            _size = iter;
        }

        /// Construct sparse vector from a dense vector
        /** \tparam S type of elements in \a v (should be castable to type \a T)
         *  \param v vector used for initialization.
         *  \param def Default value to used for the constructed sparse vector.
         *  \param sizeHint For efficiency, the number of nondefault entries can be speficied by \a sizeHint.
         */
        template <typename S>
        spvector( const std::vector<S> &v, T def=T(), size_t sizeHint=0 ) : _p(), _size(v.size()), _def(def) {
            if( sizeHint )
                reserve( sizeHint );
            for( size_t i = 0; i < v.size(); i++ )
                if( v[i] != def )
                    push_back( i, v[i] );
        }
    //@}

    /// @name Iterator interface (with respect to nondefault entries)
    //@{
        /// Constant iterator over the elements
        typedef typename nondefaults_type::const_iterator const_iterator;
        /// Iterator over the elements
        typedef typename nondefaults_type::iterator iterator;
        /// Constant reverse iterator over the elements
        typedef typename nondefaults_type::const_reverse_iterator const_reverse_iterator;
        /// Reverse iterator over the elements
        typedef typename nondefaults_type::reverse_iterator reverse_iterator;

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

    /// \name Interface similar to std::vector, but working only on default entries
    //@{
        /// Reserve memory for \a n nondefault entries
        void reserve( size_t n ) { _p.reserve( n ); }

        /// Returns size, the number of (default and nondefault) entries of this vector
        size_t size() const { return _size; }

        /// Sets allowed index range to 0..n-1 and throws away default entries whose indices became invalid
        void resize( size_t n ) {
            _size = n;
            if( _p.size() ) {
                if( _p.back().first >= n ) {
                    iterator it = lower_bound( _p.begin(), _p.end(), std::make_pair(n, T()), first_less<size_t, T>() );
                    _p.resize( distance( _p.begin(), it ) );
                }
            }
        }

        /// Erases the element pointed to by \a position and returns an iterator pointing to the next element
        iterator erase( iterator position ) {
            return _p.erase( position );
        }
    //@}

        /// Adds a new nondefault element, assuming that its index is higher than any existing index
        void push_back( size_t idx, T val ) { _p.push_back( std::make_pair( idx, val ) ); }

        /// Sets default value
        void setDefault( T def ) { _def = def; }

        /// Gets default value
        T getDefault() const { return _def; }

        /// Clears all nondefault values
        void clearNonDef() { _p.clear(); }
        
        /// Sets \a i 'th entry to \a val
        void set( size_t i, T val ) {
            DAI_DEBASSERT( i < _size );
            iterator it = lower_bound( _p.begin(), _p.end(), std::make_pair(i, T()), first_less<size_t, T>() );
            if( (it != _p.end()) && (it->first == i) ) {
                // nondefault value already present
                if( val == _def )
                    _p.erase( it );
                else
                    it->second = val;
            } else {
                // no nondefault value present yet
                if( val == _def )
                    ; // do nothing
                else
                    _p.insert( it, std::make_pair(i, val) );
            }
        }

        /// Gets \a i 'th entry
        T get( size_t i ) const {
            DAI_DEBASSERT( i < _size );
            const_iterator it = lower_bound( _p.begin(), _p.end(), std::make_pair(i, T()), first_less<size_t, T>() );
            if( (it != _p.end()) && (it->first == i) )
                return it->second;
            else
                return _def;
        }

        /// Returns a copy of the \a i 'th entry
        T operator[]( size_t i ) const { return get(i); }

        /// Returns number of nondefault values
        size_t nrNonDef() const { return _p.size(); }

        /// Returns number of default values
        size_t nrDef() const { return _size - _p.size(); }

        /// Returns default value
        T def() const { return _def; }

        /// Sets default value
        void setDef( T def ) { _def = def; }

        /// Returns a constant reference to the nondefault entries
        const nondefaults_type & nonDef() const { return _p; }
};


/// Writes a \c spvector<> to a \c std::ostream
template<class T>
std::ostream& operator << (std::ostream& os, const spvector<T> &x) {
    os << "(";
    os << "def:" << x.def();
    for( typename spvector<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << ", " << it->first << ":" << it->second;
    os << ")";
    return os;
}


} // end of namespace dai


#endif
