import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
from itertools import permutations
import math
import scipy.stats
import pandas as pd
import seaborn as sns

class Game:
    def __init__(self, n, bid_levs, utils):
        self.n = n
        self.m = bid_levs
        self.bid_levs = bid_levs
        self.A_i = np.array(list(range(bid_levs))) # possible actions for center bid, not normalized (so for 3 it's 0, 1, 2)
        self.utils = utils
        
        prefs = scipy.stats.rankdata(self.utils, axis = 1)- 1 # 0 through n-1, rank n is highest value
        self.base_bid = prefs 
        self.mids = np.where(self.base_bid == 1)[1] # where the middle pref papers are in the util mtx
        self.base_bid[self.base_bid == (self.n-1)] = self.bid_levs - 1 # adjusting top rank to be max bid
        self.A = np.array(list(it.product(self.A_i, self.A_i, self.A_i))) # all possible outcomes -- the 3 mid bids
        self.pure_utils()

    def solve(self, full_bid):
        '''given a complete bid matrix, it will return expected utils for all reviewers
            inputs:
                - full_bid: n x n matrix with entries in self.bid_levs
            outputs:
                - final_utils: n matrix with expected util for each reviewer'''
        # goes through all possible perfect matchings
        assignments = [list(tup) for tup in permutations(range(self.n), self.n)]
        matches = np.zeros((self.n, self.n)) # reviewers x papers
        max_bid_sum = 0
        # loops through possible matches and finds ones with maximum bid sum
        for a in assignments:
            bid_sum = 0
            for i in range(self.n):
                bid_sum += full_bid[i][a[i]]
            if bid_sum == max_bid_sum:
                for j in range(self.n):
                    matches[j, a[j]] = matches[j, a[j]] + 1
            if bid_sum > max_bid_sum:
                max_bid_sum = bid_sum
                matches.fill(0)
                for j in range(self.n):
                    matches[j, a[j]] = 1

        # goes through each reviewer and averages their utility in each 
        final_utils = np.zeros((self.n))
        for rev in range(self.n):
            expected_util_num = 0.0
            expected_util_denom = np.sum(matches[rev])
            for paper in range(self.n):
                expected_util_num += (matches[rev][paper] * self.utils[rev][paper])
            final_utils[rev] = expected_util_num/expected_util_denom
        return final_utils

    def pure_utils(self):
        '''creates matrix of utils for each possible pure outcome
            "outputs":
                - self.pure_strat_utils: bid levs x bid levs x ... x bid levs (n times) x n, has reviewer util for each possible bid mtx'''
        pure_shape = tuple([self.bid_levs] * self.n) + tuple([self.n])
        pure_strat_utils = np.zeros(pure_shape)
        # iterating through pure strategy space
        for mid_bid in self.A:
            # updating base bid with mid bid values
            bid = np.copy(self.base_bid)
            for i, j in enumerate(self.mids): # self.mids has indices of where middle is
                bid[i][j] = mid_bid[i]
            resulting_utils = self.solve(bid)
            pure_strat_utils[mid_bid[0]][mid_bid[1]][mid_bid[2]][:] = resulting_utils 
        self.pure_strat_utils = pure_strat_utils

    def find_M_i_star(self, rev):
        # using eq in paper for upper bound of gradient of regret 
        maximum = np.max(self.pure_strat_utils[:,:,:,rev]) # max util possible for rev
        minimum = np.min(self.pure_strat_utils[:,:,:,rev])
        return 2*(maximum - minimum)

    def get_util(self, p):
        '''Given a (potentially mixed) strategy, returns an array of utils for all reviewers
            inputs:
                - p: an element of P:  a n by m prob vec. 
            outputs:
                - expected_util: an array of utils for all reviewers'''
        expected_util = np.zeros((self.n))
        for a in self.A:
            # a is a n vector of mid bid values
            prob_a = 1
            for rev, mid_bid in enumerate(a):
                prob_a *= p[rev][mid_bid] # getting probability of a occurring
            util_a = self.pure_strat_utils[a[0]][a[1]][a[2]] # a vector of size n
            expected_util += util_a * prob_a
        return np.round(expected_util, 5)

    def find_p0_d(self, reg):
        '''Finds p0 and d of a given region
            inputs:
                - reg: n x (m-1) tuples of (lb, ub) floats dictating the hyperrectangle boundaries
            outputs:
                - (p0, d):
                    - p0: has dim n x m and each row sums to 1 (if possible), in the center of reg
                    - d: "diameter" of hyperrectangle
        '''
        p0 = []
        d_rad_sum = 0
        for rev in range(self.n):
            rev_strat = [] # building up strategy one reviewer at a time
            for dim in range(self.bid_levs-1):
                (lb, ub) = reg[rev][dim]
                rev_strat += [(lb+ub)/2.0]
                d_rad_sum += (ub-lb)**2
            # need to deal with last element
            curr_sum = np.sum(rev_strat)
            if curr_sum > 1:
                # p0 not feasible, so doesn't matter anyways 
                rev_strat += [0.0]
            else:
                rev_strat += [1.0 - curr_sum] # otherwise, make it a legit prob vector
            p0.append(rev_strat)
        d = 0.5 * math.sqrt(d_rad_sum)
        return (np.array(p0), d)
    
    def pure_action(self, action):
        '''inputs:
                - action: a bid value (for the middle paper)
            outputs:
                - pure_action: an array with m values (goes through all the actions) representing a pure action'''
        p_i = np.zeros((self.bid_levs))
        for act in self.A_i:
            if act == action:
                p_i[act] = 1.0
            else:
                p_i[act] = 0.0
        return p_i

    def get_action_regret(self, rev, action, p):
        '''inputs:
                - rev: int, reviewer number
                - action: int, bid value
                - p: an element of P -- a n by m prob vec. An array of n dicts.
            outputs:
                - regret: int, regret for reviewer i of playing p_i instead of pure strategy action 
        '''
        p_alt = np.copy(p)
        p_alt[rev] = self.pure_action(action) # all other reviewer actions stay the same 
        return self.get_util(p_alt)[rev] - self.get_util(p)[rev]
    
    def get_max_regret_rev_i(self, rev, p, v=False):
        max_action_regret = -1
        maximizing_action = []
        for action in self.A_i:
            reg = self.get_action_regret(rev, action, p)
            if reg == max_action_regret:
                maximizing_action += [action]
            if reg > max_action_regret:
                max_action_regret = reg
                maximizing_action = [action]
        return (max_action_regret, maximizing_action)

    def get_game_regret(self, p):
        max_rev_regret = 0
        for rev in range(self.n):
            reg = self.get_max_regret_rev_i(rev, p)[0]
            if reg > max_rev_regret:
                max_rev_regret = reg
        return max_rev_regret

    def M_value(self, rev, p0, A_star):
        grad_r = np.zeros((len(A_star), self.n, self.m))
        p = np.copy(p0)[:, :-1] 
        m = self.m
        for i in range(self.n): # reviewer
            for j in range(self.bid_levs): # bid level action                    
                if i == 0:
                    modified_u = (self.pure_strat_utils[j, :, :, rev] - self.pure_strat_utils[m-1, :, :, rev])
                    term_1 = np.dot(p[1], (np.dot(modified_u[:-1, :-1], p[2])))
                    term_2 = p0[2][m-1] * np.dot(modified_u[:-1, m-1], p[1])
                    term_3 = p0[1][m-1] * np.dot(modified_u[m-1, :-1], p[2])
                    term_4 = modified_u[m-1, m-1] * p0[1][m-1] * p0[2][m-1]
                if i == 1:
                    modified_u = (self.pure_strat_utils[:, j, :, rev] - self.pure_strat_utils[:, m-1, :, rev])
                    term_1 = np.dot(p[0], (np.dot(modified_u[:-1, :-1], p[2])))
                    term_2 = p0[2][m-1] * np.dot(modified_u[:-1, m-1], p[0])
                    term_3 = p0[0][m-1] * np.dot(modified_u[m-1, :-1], p[2])
                    term_4 = modified_u[m-1, m-1] * p0[0][m-1] * p0[2][m-1]
                else:
                    modified_u = (self.pure_strat_utils[:, :, j, rev] - self.pure_strat_utils[:, :, m-1, rev])
                    term_1 = np.dot(p[0], (np.dot(modified_u[:-1, :-1], p[2])))
                    term_2 = p0[1][m-1] * np.dot(modified_u[:-1, m-1], p[0])
                    term_3 = p0[0][m-1] * np.dot(modified_u[m-1, :-1], p[1])
                    term_4 = modified_u[m-1, m-1] * p0[0][m-1] * p0[1][m-1]
                
                grad_u_i = term_1 + term_2 + term_3 + term_4

                for ind, action in enumerate(A_star):
                    if rev == i:
                        grad_u_i_a = 0
                    else: 
                        util_fixed_action_rev = np.take(self.pure_strat_utils, action, axis=rev)[:, :, rev]
                        if i < rev:
                            util_fixed_i_j = np.take(util_fixed_action_rev, j, axis = i)
                            util_fixed_i_m = np.take(util_fixed_action_rev, m-1, axis = i)
                        else:
                            util_fixed_i_j = np.take(util_fixed_action_rev, j, axis = i-1)
                            util_fixed_i_m = np.take(util_fixed_action_rev, m-1, axis = i-1)
                        idxs = {0, 1, 2}
                        idxs.remove(rev)
                        idxs.remove(i)
                        remaining_rev_axis = idxs.pop()
                        term_1 = np.dot((util_fixed_i_j - util_fixed_i_m)[:-1], p[remaining_rev_axis])
                        term_2 = (util_fixed_i_j[m-1] - util_fixed_i_m[m-1]) * p0[remaining_rev_axis][m-1]
                        grad_u_i_a = term_1+term_2
                    grad_r[ind][i][j] = grad_u_i_a - grad_u_i
        return np.max(grad_r)

    def g(self, region, p0, d):
        max_ratio = -1
        for rev in range(self.n):
            (reg, A_star) = self.get_max_regret_rev_i(rev, p0)
            M_i = self.M_value(rev, p0, A_star)
            if reg == 0:
                val = 0
            else:
                val = reg/(d * (M_i + np.random.uniform(0.00001, 0.00002, 1)[0]))
            if val > max_ratio:
                max_ratio = val
        return max_ratio

def region_outside_space(region):
    all_lbs = np.array([[dim[0] for dim in rev] for rev in region])
    return (np.sum(all_lbs, axis=1) > 1).any()

def run_exclusion(n, eps, bid_levs, utils, num_nes):
    game = Game(n, bid_levs, utils)
    m = bid_levs
    # regions will be in the form [[(l_{p_i(a_j)}, u_{p_i(a_j)}) for j in 1, ..., m-1] for i in 1, ..., n]
    # so have dimension m * n, and each dimension has lower bound and upper bound
    regions = {}
    P_init = [[(0.0, 1.0) for j in range(m-1)] for i in range(n)] 

    # initialize region -- first will be just all of space
    (p0, d) = game.find_p0_d(P_init)
    init_g = game.g(P_init, p0, d)
    regions[init_g] = (P_init, p0, d) 
    
    # compute M_i*
    M_i_star = np.zeros((n))
    for i in range(n):
        M_i_star[i] = game.find_M_i_star(i)
    
    r_p0s = []
    p0s = []
    for ne in range(num_nes):
        r_p0 = 500
        counter = 1
        min_r_p0 = 500
        min_p0 = []
        if len(regions) == 0:
            print("no more regions")
            break
        # repeat until r(p) small enough:
        while r_p0 > eps and counter < 3000:
            # 1. select region with minimal value of g in Eq (2)
            if len(regions) == 0:
                print("no more regions")
                break
            g = min(regions.keys())
            min_g_reg = regions.pop(g)
            # 2. select p0. If it is outside the search space, goto 3.
            (reg, p0, d) = min_g_reg
            if (np.sum(p0, axis=1) <= 1.001).all():
                #inside search space
                # Else compute r(p0). If Eq (4) satisfied, exclude the region and return to 1.
                r_p0_rev = np.zeros((n))
                for i in range(n):
                    r_p0_rev[i] = game.get_max_regret_rev_i(i, p0)[0]
                r_p0 = np.max(r_p0_rev)
                if r_p0 < min_r_p0:
                    min_r_p0 = r_p0
                    min_p0 = p0
                if (r_p0_rev > (d * M_i_star)).any():
                    continue
                    # can exclude region

            # Else, goto 3.
            # 3. Bisect the region and compute g for the new regions with r(p0) and M_i(p0) in Eq (3)
            max_edge_len = 0
            max_ind = (0,0)
            for i in range(n):
                for j in range(m-1):
                    (lb, ub) = reg[i][j]
                    edge_len = ub-lb
                    if edge_len > max_edge_len:
                        max_ind = (i, j)
                        max_edge_len = edge_len
            (old_lb, old_ub) = reg[max_ind[0]][max_ind[1]]
            reg_upper_half = np.copy(reg)
            reg_upper_half[max_ind[0]][max_ind[1]] = (old_lb + 0.5 * max_edge_len, old_ub)
            if not region_outside_space(reg_upper_half):
                (reg_up, d_up) = game.find_p0_d(reg_upper_half)
                if (np.sum(reg_up, axis=1) <= 1).all():
                    reg_up_g = game.g(reg_upper_half, reg_up, d_up)
                else:
                    reg_up_g = 2 * g + np.random.uniform(0.00001, 0.00002, 1)[0]
                if r_p0 <= eps:
                    reg_up_g *= 10
                regions[reg_up_g] = (reg_upper_half, reg_up, d_up)

            reg_lower_half = np.copy(reg)
            reg_lower_half[max_ind[0]][max_ind[1]] = (old_lb, old_lb + 0.5 * max_edge_len)
            if not region_outside_space(reg_lower_half):    
                (reg_low, d_low) = game.find_p0_d(reg_lower_half)
                if (np.sum(reg_low, axis=1) <= 1).all():
                    reg_low_g = game.g(reg_lower_half, reg_low, d_low)
                else: 
                    reg_low_g = 2 * g + np.random.uniform(0.00001, 0.00002, 1)[0]
                if r_p0 <= eps:
                    reg_low_g *= 10
                regions[reg_low_g] = (reg_lower_half, reg_low, d_low)
            counter += 1
        p0s.append(min_p0)
        r_p0s.append(min_r_p0)
        # print("UTILS: ", game.utils)
        # print("FINAL NE:", min_p0, "REGRET: ", min_r_p0)
    return p0s, np.array([game.get_util(min_p0) for min_p0 in p0s])

def main():
    reps = 9
    num_nes = 15
    ne_utils = np.zeros((2, reps, num_nes))
    min_2 = np.zeros((reps))
    max_2 = np.zeros((reps))
    min_3 = np.zeros((reps))
    max_3 = np.zeros((reps))

    utils = np.array([[1, -1.0, 0], [1, 0.5, 0.], [1.0, 0.0, 0.5]])
    for i in range(reps):
        print("rep: ", i)
        utils[0][1] = (i+1)/((reps+1)*1.0)
        print("utils: ", utils)
        (p0_2, p0_2_u) = run_exclusion(3, 0.001, 2, utils, num_nes)
        ne_utils[0][i] = np.average(p0_2_u, axis=1)
        print("utils for 2:", p0_2_u)
        (p0_3, p0_3_u) = run_exclusion(3, 0.001, 3, utils, num_nes)
        ne_utils[1][i] = np.average(p0_3_u, axis=1)
        print("utils for 3:", p0_3_u)
    xticks = [(i+1) for i in range(reps)]
    mid_utils = [round((i+1)/((reps+1)*1.0), 2) for i in range(reps)]
    xs = list(range(reps))

    data = {'2 Bid Levels': np.ndarray.flatten(ne_utils[0]), '3 Bid Levels': np.ndarray.flatten(ne_utils[1]), 
    'Changing Utility Value': sorted(mid_utils * num_nes)}
    df = pd.DataFrame(data)
    dfl = pd.melt(df, id_vars='Changing Utility Value', value_vars = ['2 Bid Levels', '3 Bid Levels'])
    sns.boxplot(x='Changing Utility Value', y='value', data=dfl, showfliers=False, color='tomato', hue='variable')
    plt.ylabel("Average Utility Per Reviewer in Equilibrium")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    title = "Overall Utility of Mixed Equilibria Under 2 vs 3 Bid Levels"
    plt.title(title)
    plt.savefig("Saturday_" + title + ".png", bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    main()


