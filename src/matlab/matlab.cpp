/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <dai/matlab/matlab.h>

namespace {

void getMemberAndP(const mxArray* psi, size_t ind,
                   mxArray** mx_member, mxArray** mx_P) {
    if( mxIsCell(psi) ) {
        mxArray* cell = mxGetCell(psi, ind);
        if( !mxIsStruct(cell) )
            mexErrMsgTxt("invalid factor psi.");
        *mx_member = mxGetField(cell, ind, "Member");
        *mx_P = mxGetField(cell, ind, "P");
    }
    else if ( mxIsStruct(psi) ) {
        *mx_member = mxGetField(psi, ind, "Member");
        *mx_P = mxGetField(psi, ind, "P");
    }
    else
        mexErrMsgTxt("invalid factor psi.");

    if (!*mx_member || !*mx_P)
        mexErrMsgTxt("could not find a 'Member' or 'P' field.");
    if (!mxIsDouble(*mx_member) || !mxIsDouble(*mx_P))
        mexErrMsgTxt("field 'Member' or 'P' is invalid.");
}

}

namespace dai {


using namespace std;


/* Convert vector<Factor> structure to a struct array of CPTAB-like structs */
mxArray *Factors2mx(const vector<Factor> &Ps) {
    size_t nr = Ps.size();

    const char *fieldnames[] = { "Member", "P" };
    mxArray *psi = mxCreateStructMatrix(nr,1,2,fieldnames);
    if( !psi )
        mexErrMsgTxt("Memory error");

    size_t I_ind = 0;
    for( vector<Factor>::const_iterator I = Ps.begin(); I != Ps.end(); I++, I_ind++ ) {
        mxArray *BiMember = mxCreateDoubleMatrix(1,I->vars().size(),mxREAL);
        if( !BiMember )
            mexErrMsgTxt("Memory error");
        double *BiMember_data = mxGetPr(BiMember);
        size_t i = 0;
        vector<mwSize> dims;
        for( VarSet::const_iterator j = I->vars().begin(); j != I->vars().end(); j++,i++ ) {
            BiMember_data[i] = j->label();
            dims.push_back( j->states() );
        }
        while( dims.size() <= 2 )
            dims.push_back( 1 );

        mxArray *BiP = mxCreateNumericArray(dims.size(), &(*(dims.begin())), mxDOUBLE_CLASS, mxREAL);
        if( !BiP )
            mexErrMsgTxt("Memory error");
        double *BiP_data = mxGetPr(BiP);
        for( size_t j = 0; j < I->nrStates(); j++ )
            BiP_data[j] = (*I)[j];

        mxSetField(psi,I_ind,"Member",BiMember);
        mxSetField(psi,I_ind,"P",BiP);
    }
    return( psi );
}

/* Convert struct array of CPTAB-like structs to vector<Factor> */
vector<Factor> mx2Factors(const mxArray *psi, long verbose) {
    set<Var> vars;
    vector<Factor> factors;
    size_t n = mxGetNumberOfElements(psi);

    for( size_t cellind = 0; cellind < n; cellind++ ) {
        if( verbose >= 3 )
            cerr << "reading factor " << cellind << ": " << endl;
        mxArray *mx_member, *mx_P;
        getMemberAndP(psi, cellind, &mx_member, &mx_P);
        size_t nr_mem = mxGetNumberOfElements(mx_member);
        double *members = mxGetPr(mx_member);
        const mwSize *dims = mxGetDimensions(mx_P);
        double *factordata = mxGetPr(mx_P);

        // add variables
        VarSet factorvars;
        vector<long> labels(nr_mem,0);
        if( verbose >= 3 )
            cerr << "  vars: ";
        for( size_t mi = 0; mi < nr_mem; mi++ ) {
            labels[mi] = (long)members[mi];
            if( verbose >= 3 )
                cerr << labels[mi] << "(" << dims[mi] << ") ";
            vars.insert( Var(labels[mi], dims[mi]) );
            factorvars |= Var(labels[mi], dims[mi]);
        }
        factors.push_back(Factor(factorvars));

        // calculate permutation matrix
        vector<size_t> perm(nr_mem,0);
        VarSet::iterator j = factorvars.begin();
        for( size_t mi = 0; mi < nr_mem; mi++,j++ ) {
            long gezocht = j->label();
            vector<long>::iterator piet = find(labels.begin(),labels.end(),gezocht);
            perm[mi] = piet - labels.begin();
        }

        if( verbose >= 3 ) {
            cerr << endl << "  perm: ";
            for( vector<size_t>::iterator r=perm.begin(); r!=perm.end(); r++ )
                cerr << *r << " ";
            cerr << endl;
        }

        // read Factor
        vector<size_t> di(nr_mem,0);
        size_t prod = 1;
        for( size_t k = 0; k < nr_mem; k++ ) {
            di[k] = dims[k];
            prod *= dims[k];
        }
        Permute permindex( di, perm );
        for( size_t li = 0; li < prod; li++ )
            factors.back().set( permindex.convertLinearIndex(li), factordata[li] );
    }

    if( verbose >= 3 ) {
        for(vector<Factor>::const_iterator I=factors.begin(); I!=factors.end(); I++ )
            cerr << *I << endl;
    }

    return( factors );
}

/* Convert CPTAB-like struct to Factor */
Factor mx2Factor(const mxArray *psi) {
    if( mxGetNumberOfDimensions(psi)!=1 )
        mexErrMsgTxt("invalid factor psi.");

    mxArray *mx_member, *mx_P;
    getMemberAndP(psi, 0, &mx_member, &mx_P);
    size_t nr_mem = mxGetN(mx_member);
    double *members = mxGetPr(mx_member);
    const mwSize *dims = mxGetDimensions(mx_P);
    double *factordata = mxGetPr(mx_P);

    // add variables
    VarSet vars;
    vector<long> labels(nr_mem,0);
    for( size_t mi = 0; mi < nr_mem; mi++ ) {
        labels[mi] = (long)members[mi];
        vars |= Var(labels[mi], dims[mi]);
    }
    Factor factor(vars);

    // calculate permutation matrix
    vector<size_t> perm(nr_mem,0);
    VarSet::iterator j = vars.begin();
    for( size_t mi = 0; mi < nr_mem; mi++,j++ ) {
        long gezocht = j->label();
        vector<long>::iterator piet = find(labels.begin(),labels.end(),gezocht);
        perm[mi] = piet - labels.begin();
    }

    // read Factor
    vector<size_t> di(nr_mem,0);
    size_t prod = 1;
    for( size_t k = 0; k < nr_mem; k++ ) {
        di[k] = dims[k];
        prod *= dims[k];
    }
    Permute permindex( di, perm );
    for( size_t li = 0; li < prod; li++ )
        factor.set( permindex.convertLinearIndex(li), factordata[li] );

    return( factor );
}


}
