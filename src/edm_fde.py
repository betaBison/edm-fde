########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         performs EDM-based Fault detection and exclusion
########################################################################

import numpy as np

def edm(X):
    """Creates a Euclidean distance matrix (EDM) from point locations.

    See [1]_ for more explanation.

    Parameters
    ----------
    X : np.array
        Locations of points/nodes in the graph. Numpy array of shape
        state space dimensions x number of points in graph.

    Returns
    -------
    D : np.array
        Euclidean distance matrix as a numpy array of shape (n x n)
        where n is the number of points in the graph.
        creates edm from points

    References
    ----------
    ..  [1] I. Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli.
        “Euclidean Distance Matrices: Essential Theory, Algorithms,
        and Applications.” 2015. arxiv.org/abs/1502.07541.

    """
    n = X.shape[1]
    G = (X.T).dot(X)
    D = np.diag(G).reshape(-1,1).dot(np.ones((1,n))) \
        - 2.*G + np.ones((n,1)).dot(np.diag(G).reshape(1,-1))
    return D

def edm_from_satellites_ranges(S,ranges):
    """Creates a Euclidean distance matrix (EDM) from points and ranges.

    Creates an EDM from a combination of known satellite positions as
    well as ranges from between the receiver and satellites.

    Parameters
    ----------
    S : np.array
        known locations of satellites packed as a numpy array in the
        shape state space dimensions x number of satellites.
    ranges : np.array
        ranges between the receiver and satellites packed as a numpy
        array in the shape 1 x number of satellites

    Returns
    -------
    D : np.array
        Euclidean distance matrix in the shape (1 + s) x (1 + s) where
        s is the number of satellites

    """
    num_s = S.shape[1]
    D = np.zeros((num_s+1,num_s+1))
    D[0,1:] = ranges**2
    D[1:,0] = ranges**2
    D[1:,1:] = edm(S)

    return D

def edm_fde(D, dims, max_faults = None, edm_threshold = 1.0,
            verbose = False):
    """Performs EDM-based fault detection and exclusion (FDE).

    See [1]_ for more detailed explanation of algorithm.

    Parameters
    ----------
    D : np.array
        Euclidean distance matrix (EDM) of shape n x n where n is the
        number of satellites + 1.
    dims : int
        Dimensions of the state space.
    max_faults : int
        Maximum number of faults to exclude (corresponds to fault
        hypothesis). If set to None, then no limit is set.
    edm_threshold : float
        EDM-based FDE thresholding parameter. For an explanation of the
        detection threshold see [1]_.
    verbose : bool
        If true, prints a variety of helpful debugging statements.

    Returns
    -------
    tri : list
        indexes that should be exluded from the measurements

    References
    ----------
    ..  [1] D. Knowles and G. Gao. "Euclidean Distance Matrix-based
        Rapid Fault Detection and Exclusion." ION GNSS+ 2021.

    """

    ri = None                   # index to remove
    tri = []                    # removed indexes (in transmitter frame)
    reci = 0                    # index of the receiver
    oi = np.arange(D.shape[0])  # original indexes

    while True:

        if ri != None:
            if verbose:
                print("removing index: ",ri)

            # add removed index to index list passed back
            tri.append(oi[ri]-1)
            # keep track of original indexes (since deleting)
            oi = np.delete(oi,ri)
            # remove index from EDM
            D = np.delete(D,ri,axis=0)
            D = np.delete(D,ri,axis=1)


        n = D.shape[0]  # shape of EDM

        # stop removing indexes either b/c you need at least four
        # satellites or if maximum number of faults has been reached
        if n <= 5 or (max_faults != None and len(tri) >= max_faults):
            break


        # double center EDM to retrive the corresponding Gram matrix
        J = np.eye(n) - (1./n)*np.ones((n,n))
        G = -0.5*J.dot(D).dot(J)

        # perform singular value decomposition
        U, S, Vh = np.linalg.svd(G)

        # calculate detection test statistic
        warn = S[dims]*(sum(S[dims:])/float(len(S[dims:])))/S[0]
        if verbose:
            print("\nDetection test statistic:",warn)

        if warn > edm_threshold:
            ri = None

            u_mins = set(np.argsort(U[:,dims])[:2])
            u_maxes = set(np.argsort(U[:,dims])[-2:])
            v_mins = set(np.argsort(Vh[dims,:])[:2])
            v_maxes = set(np.argsort(Vh[dims,:])[-2:])

            def test_option(ri_option):
                # remove option
                D_opt = np.delete(D.copy(),ri_option,axis=0)
                D_opt = np.delete(D_opt,ri_option,axis=1)

                # reperform double centering to obtain Gram matrix
                n_opt = D_opt.shape[0]
                J_opt = np.eye(n_opt) - (1./n_opt)*np.ones((n_opt,n_opt))
                G_opt = -0.5*J_opt.dot(D_opt).dot(J_opt)

                # perform singular value decomposition
                _, S_opt, _ = np.linalg.svd(G_opt)

                # calculate detection test statistic
                warn_opt = S_opt[dims]*(sum(S_opt[dims:])/float(len(S_opt[dims:])))/S_opt[0]

                return warn_opt


            # get all potential options
            ri_options = u_mins | v_mins | u_maxes | v_maxes
            # remove the receiver as a potential fault
            ri_options = ri_options - set([reci])
            ri_tested = []
            ri_warns = []

            ui = -1
            while np.argsort(np.abs(U[:,dims]))[ui] in ri_options:
                ri_option = np.argsort(np.abs(U[:,dims]))[ui]

                # calculate test statistic after removing index
                warn_opt = test_option(ri_option)

                # break if test statistic decreased below threshold
                if warn_opt < edm_threshold:
                    ri = ri_option
                    if verbose:
                        print("chosen ri: ", ri)
                    break
                else:
                    ri_tested.append(ri_option)
                    ri_warns.append(warn_opt)
                ui -= 1

            # continue searching set if didn't find index
            if ri == None:
                ri_options_left = list(ri_options - set(ri_tested))

                for ri_option in ri_options_left:
                    warn_opt = test_option(ri_option)

                    if warn_opt < edm_threshold:
                        ri = ri_option
                        if verbose:
                            print("chosen ri: ", ri)
                        break
                    else:
                        ri_tested.append(ri_option)
                        ri_warns.append(warn_opt)

            # if no faults decreased below threshold, then remove the
            # index corresponding to the lowest test statistic value
            if ri == None:
                idx_best = np.argmin(np.array(ri_warns))
                ri = ri_tested[idx_best]
                if verbose:
                    print("chosen ri: ", ri)

        else:
            break

    return tri
