import numpy as np
import os
import pickle

"""
%%%%%%%%%%%%%%%%%%%%%%% Classes to store results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class TestResults:
    
    def __init__(self, name, B):
        self.name = name
        self.B = B
        self.rejects = 0
        self.estimator_values = np.zeros(B+1)
        self.unsorted_estimator_values = np.zeros(B+1)
        self.statistic_values = 0
        self.threshold_values = 0
        self.times = np.zeros(B+1)
        self.total_times = 0
        self.compute = True
        self.file_exists = None
        self.fname = None
        
    def set_compute(self, no_compute, recompute):
        if no_compute:
            self.compute = False
        elif recompute:
            self.compute = True
        elif self.file_exists:
            self.compute = False
        else:
            self.compute = True
        
    def set_statistic_value(self):
        self.statistic_values = self.estimator_values[self.B]
        
    def sort_estimator_values(self):
        self.unsorted_estimator_values[:] = self.estimator_values[:]
        self.estimator_values[:] = np.sort(self.estimator_values[:])
        
    def set_threshold(self, alpha):
        self.threshold_values = self.estimator_values[np.ceil((1-alpha)*(self.B+1)).astype('int')-1] 
        #...-1 because indices go from 0 to B instead of 1 to B+1
        
    def set_reject(self):
        if self.threshold_values == 0:
            self.rejects = 0
        elif self.statistic_values > self.threshold_values:
            self.rejects = 1
        else:
            self.rejects = 0
            
    def set_total_times(self,sum_across_bw=True):
        self.total_times = np.sum(self.times[:])
        
    def save(self):
        pickle.dump(self, open(self.fname, 'wb'))
        
        
class AggregatedTestResults:
    
    def __init__(self, name, B, B_2, n_bandwidths, bw, weights_vec):
        self.name = name
        self.B = B
        self.B_2 = B_2
        self.rejects = 0
        self.rejects_median = 0
        self.n_bandwidths = n_bandwidths
        self.bw = bw
        self.weights_vec = weights_vec
        self.hat_u_alpha = None
        self.all_estimator_values = dict()
        self.estimator_values = dict()
        self.estimator_values_2 = dict()
        self.statistic_values = dict()
        self.threshold_values = dict()
        self.times = dict()
        self.times_2 = dict()
        for bandwidth in self.bw:
            self.all_estimator_values[bandwidth] = np.zeros(B_2+B+1)
            self.estimator_values[bandwidth] = np.zeros(B+1)
            self.estimator_values_2[bandwidth] = np.zeros(B_2)
            self.statistic_values[bandwidth] = 0
            self.threshold_values[bandwidth] = 0
            self.times[bandwidth] = np.zeros(B+1)
            self.times_2[bandwidth] = np.zeros(B_2)
        self.total_times = 0
        self.compute = True
        self.file_exists = None
        self.fname = None
        
    def set_compute(self, no_compute, recompute):
        if no_compute:
            self.compute = False
        elif recompute:
            self.compute = True
        elif self.file_exists:
            self.compute = False
        else:
            self.compute = True    
        
    def split_tests(self):
        for bandwidth in self.bw:
            self.estimator_values_2[bandwidth] = self.all_estimator_values[bandwidth][:self.B_2]
            self.estimator_values[bandwidth] = self.all_estimator_values[bandwidth][self.B_2:]
        
    def set_statistic_value(self):
        for bandwidth in self.bw:
            self.statistic_values[bandwidth] = self.estimator_values[bandwidth][self.B]
        
    def sort_estimator_values(self):
        for bandwidth in self.bw:
            self.estimator_values[bandwidth][:] = np.sort(self.estimator_values[bandwidth][:])
        
    def set_threshold(self, alpha):
        for bandwidth in self.bw:
            threshold_position = np.ceil((1-alpha)*(self.B+1)).astype('int')
            print(f'threshold_position: {threshold_position}')
            self.threshold_values[bandwidth] = self.estimator_values[bandwidth][np.ceil((1-alpha)*(self.B+1)).astype('int')-1] 
        #...-1 because indices go from 0 to B instead of 1 to B+1
        
    def set_reject(self):
        quantile = np.ceil((self.B+1)*(1-self.hat_u_alpha*self.weights_vec)).astype(int)
        print(f'self.hat_u_alpha: {self.hat_u_alpha}, self.weights_vec: {self.weights_vec}, quantile: {quantile}, B+1: {self.B + 1}.')
        self.rejects = 0
        for i in range(self.n_bandwidths):
            self.threshold_values[self.bw[i]] = self.estimator_values[self.bw[i]][quantile[i]-1]
            if self.statistic_values[self.bw[i]] > self.estimator_values[self.bw[i]][quantile[i]-1]:
                self.rejects = 1
            
    def set_total_times(self,sum_across_bw=True):
        if sum_across_bw:
            self.total_times = 0
            for bandwidth in self.bw:
                self.total_times += np.sum(self.times[bandwidth][:]) + np.sum(self.times_2[bandwidth][:])
        else:
            self.total_times = np.sum(self.times)     
        
    def compute_hat_u_alpha(self, B_3, alpha):
        u_min = 0
        u_max = 1/np.max(self.weights_vec)
        for k in range(B_3):
            u = (u_min+u_max)/2
            P_u = 0
            quantile = np.ceil((self.B+1)*(1-u*self.weights_vec)).astype(int)
            for m in range(self.B_2):
                indicator_max_value = 0
                for i in range(self.n_bandwidths):
                    if self.estimator_values_2[self.bw[i]][m] - self.estimator_values[self.bw[i]][quantile[i]-1] > 0:
                        indicator_max_value = 1
                        break
                P_u += indicator_max_value
            print(f'P_u (before dividing): {P_u}. self.B_2: {self.B_2}. u: {u}. self.weights_vec: {self.weights_vec}. quantile: {quantile}.')
            P_u = P_u/self.B_2
            if P_u <= alpha:
                u_min = u
            else:
                u_max = u
            print(f'P_u = {P_u}, alpha = {alpha}')
            print(f'k={k}: u_min={u_min}')
        self.hat_u_alpha = u_min
        
    def save(self):
        pickle.dump(self, open(self.fname, 'wb'))
        
    def set_reject_median(self):
        if self.statistic_values[self.bw[self.n_bandwidths-1]] > self.threshold_values[self.bw[self.n_bandwidths-1]]:
            self.rejects_median = 1
        else:
            self.rejects_median = 0
            

class GroupResults:
    
    def __init__(self, n_tests, B, fname_group, file_exists_group, n_bandwidths, bw, B_2 = 0, weights_vec = None):
        self.group_names = None
        self.full_group_names = None
        self.group_labels = None
        self.B = B
        self.B_2 = B_2
        self.n_bandwidths = n_bandwidths
        self.bw = bw
        self.weights_vec = weights_vec
        self.fname = None
        self.file_exists = None
        self.fname_group = fname_group
        self.file_exists_group = file_exists_group
        self.compute = dict()
        self.compute_group = True
        self.group_tests = dict()
        self.rejects = dict()
        self.rejects_median = dict()
        self.statistic_values = dict()
        self.threshold_values = dict()
        self.times = dict()
        
    #Set group names    
    def set_group_names(self, group_names, full_group_names, group_labels):
        self.group_names = group_names
        self.full_group_names = full_group_names
        self.group_labels = group_labels
        for name in group_names:
            if self.n_bandwidths == 1:
                self.group_tests[name] = TestResults(name, self.B)
            else:
                self.group_tests[name] = AggregatedTestResults(name, self.B, self.B_2, self.n_bandwidths, self.bw, self.weights_vec)
        
    #Set compute attribute
    def set_compute_group(self, no_compute, recompute):
        if no_compute:
            self.compute_group = False
        elif recompute:
            self.compute_group = True
        elif self.file_exists_group:
            self.compute_group = False
        else:
            self.compute_group = True
            
    def set_compute(self, no_compute, recompute, fname, file_exists):
        for name in self.group_names:
            self.group_tests[name].file_exists = file_exists[name]
            self.group_tests[name].fname = fname[name]
            self.group_tests[name].set_compute(no_compute, recompute)
            print(f'{name}: file_exists = {self.group_tests[name].file_exists}, compute = {self.group_tests[name].compute}, fname = {self.group_tests[name].fname}')
            
    def split_tests(self):
        for name in self.group_names:
            self.group_tests[name].split_tests()
            
    def set_statistic_value(self):
        for name in self.group_names:
            self.group_tests[name].set_statistic_value()
            
    def sort_estimator_values(self):
        for name in self.group_names:
            self.group_tests[name].sort_estimator_values()
            
    def set_threshold(self, alpha):
        for name in self.group_names:
            self.group_tests[name].set_threshold(alpha)
            
    def set_reject(self):
        for name in self.group_names:
            self.group_tests[name].set_reject()
            
    def set_reject_median(self):
        for name in self.group_names:
            self.group_tests[name].set_reject_median()
            
    def set_total_times(self,sum_across_bw=True):
        for name in self.group_names:
            self.group_tests[name].set_total_times(sum_across_bw)
        
    def set_group_results(self):
        #print(f'get_group_results. self.n_bandwidths: {self.n_bandwidths}')
        if self.n_bandwidths == 1:
            for name in self.group_names:
                self.rejects[name] = self.group_tests[name].rejects
                self.statistic_values[name] = self.group_tests[name].statistic_values
                self.threshold_values[name] = self.group_tests[name].threshold_values
                self.times[name] = self.group_tests[name].total_times
        else:
            for name in self.group_names:
                self.rejects[name] = self.group_tests[name].rejects
                self.rejects_median[name] = self.group_tests[name].rejects_median
                self.times[name] = self.group_tests[name].total_times
                self.statistic_values[name] = dict()
                self.threshold_values[name] = dict()
                for bandwidth in self.bw:
                    self.statistic_values[name][bandwidth] = self.group_tests[name].statistic_values[bandwidth]
                    self.threshold_values[name][bandwidth] = self.group_tests[name].threshold_values[bandwidth]
                #print(f'self.statistic_values[name].keys(): {self.statistic_values[name].keys()}')
    
    def compute_hat_u_alpha(self, B_3, alpha):
        for name in self.group_names:
            self.group_tests[name].compute_hat_u_alpha(B_3, alpha)
        
    def save_results(self, args):
        #store results
        res = {
            'rejects': self.rejects,
            'rejects_median': self.rejects_median,
            'statistic_values': self.statistic_values,
            'threshold_values': self.threshold_values,
            'times': self.times,
            'group_names': self.group_names,
            'full_group_names': self.full_group_names,
            'group_labels': self.group_labels,
            'bw': self.bw,
        }
        
        print(f'save_results: {self.group_names}')
        print(f'file name: {self.fname_group}')

        if not args.interactive:
            pickle.dump(res, open(self.fname_group, 'wb'))
    
    def save_objects(self, args):
        if not args.interactive:
            for name in self.group_names:
                self.group_tests[name].save()
            
        #store each Test_Results object also
        
            
class JointGroupResults:
    
    def __init__(self, fname, total_n_tests, B, n_bandwidths, file_exists):
        self.group_names = None
        self.full_group_names = None
        self.group_labels = None
        self.total_n_tests = total_n_tests
        self.B = B
        self.n_bandwidths = n_bandwidths
        self.bw = None
        self.fname = fname
        self.file_exists = file_exists
        self.compute = False
        
        self.rejects = dict()
        self.rejects_median = dict()
        self.statistic_values = dict()
        self.threshold_values = dict()
        self.times = dict()
        
        self.rejection_rates = dict()
        self.rejection_rates_median = dict()
        self.statistic_means = dict()
        self.statistic_stds = dict()
        self.threshold_means = dict()
        self.threshold_stds = dict()
        self.time_means = dict()
        
    #Set compute attribute
    def set_compute(self, no_compute):
        if no_compute:
            self.compute = False
        elif not self.file_exists:
            self.compute = False
        else:
            self.compute = True
            
    #Set group names    
    def set_group_names(self, group_names, full_group_names, group_labels):
        self.group_names = group_names
        self.full_group_names = full_group_names
        self.group_labels = group_labels
        
    #Set bandwidths and weights_vec
    def set_bandwidth(self, bw):
        self.bw = bw
            
    def update_attributes(self, res):
        for name in self.group_names:
            
            if self.n_bandwidths == 1:
                if name not in self.statistic_values.keys():
                    self.statistic_values[name] = [res['statistic_values'][name]]
                else:
                    self.statistic_values[name].append(res['statistic_values'][name])

                if name not in self.threshold_values.keys():
                    self.threshold_values[name] = [res['threshold_values'][name]]
                else:
                    self.threshold_values[name].append(res['threshold_values'][name])
            else:
                if name not in self.statistic_values.keys():
                    self.statistic_values[name] = dict()
                    for bandwidth in self.bw:
                        self.statistic_values[name][bandwidth] = [res['statistic_values'][name][bandwidth]]
                else:
                    for bandwidth in self.bw:
                        self.statistic_values[name][bandwidth].append(res['statistic_values'][name][bandwidth])
                        
                if name not in self.threshold_values.keys():
                    self.threshold_values[name] = dict()
                    for bandwidth in self.bw:
                        self.threshold_values[name][bandwidth] = [res['threshold_values'][name][bandwidth]]
                else:
                    for bandwidth in self.bw:
                        self.threshold_values[name][bandwidth].append(res['threshold_values'][name][bandwidth])
                
                if name not in self.rejects_median.keys():
                    self.rejects_median[name] = [res['rejects_median'][name]]
                else:
                    self.rejects_median[name].append(res['rejects_median'][name])
            
            if name not in self.rejects.keys():
                self.rejects[name] = [res['rejects'][name]]
            else:
                self.rejects[name].append(res['rejects'][name])
            
            if name not in self.times.keys():
                self.times[name] = [res['times'][name]]
            else:
                self.times[name].append(res['times'][name])
                
            #res_times_name = res['times'][name]
            #print(f'res_times_name: {res_times_name}')
                
            self.bw = res['bw']
            
    def compute_info(self):
        #print(f'self.n_bandwidths: {self.n_bandwidths}')
        if self.n_bandwidths == 1:
            for name in self.group_names:
                self.rejection_rates[name] = np.mean(self.rejects[name])
                self.statistic_means[name] = np.mean(self.statistic_values[name])
                self.statistic_stds[name] = np.std(self.statistic_values[name])
                self.threshold_means[name] = np.mean(self.threshold_values[name])
                self.threshold_stds[name] = np.std(self.threshold_values[name])
                self.time_means[name] = np.mean(self.times[name])
        else:
            for name in self.group_names:
                self.statistic_means[name] = dict()
                self.statistic_stds[name] = dict()
                self.threshold_means[name] = dict()
                self.threshold_stds[name] = dict()
                for bandwidth in self.bw:
                    #print(f'len(self.statistic_values): {len(self.statistic_values)}')
                    #print(f'len(self.statistic_values[name]): {len(self.statistic_values[name])}')
                    #print(f'self.statistic_values[name][bandwidth]: {self.statistic_values[name][bandwidth]}')
                    self.statistic_means[name][bandwidth] = np.mean(np.array(self.statistic_values[name][bandwidth]))
                    self.statistic_stds[name][bandwidth] = np.std(np.array(self.statistic_values[name][bandwidth]))
                    self.threshold_means[name][bandwidth] = np.mean(np.array(self.threshold_values[name][bandwidth]))
                    self.threshold_stds[name][bandwidth] = np.std(np.array(self.threshold_values[name][bandwidth]))
                self.rejection_rates[name] = np.mean(self.rejects[name])
                self.rejection_rates_median[name] = np.mean(self.rejects_median[name])
                self.time_means[name] = np.mean(self.times[name])
                
    def print_info(self):
        for i in range(len(self.group_names)):
            print(f'Rejection rate {self.full_group_names[i]}: {self.rejection_rates[self.group_names[i]]}.')
        if self.n_bandwidths != 1:
            for i in range(len(self.group_names)):
                print(f'Rejection rate median {self.full_group_names[i]}: {self.rejection_rates_median[self.group_names[i]]}.')
        for i in range(len(self.group_names)):
            print(f'Statistic mean {self.full_group_names[i]}: {self.statistic_means[self.group_names[i]]}. Statistic std. {self.full_group_names[i]}: {self.statistic_stds[self.group_names[i]]}.')
        for i in range(len(self.group_names)):
            print(f'Threshold mean {self.full_group_names[i]}: {self.threshold_means[self.group_names[i]]}. Threshold std. {self.full_group_names[i]}: {self.threshold_stds[self.group_names[i]]}.')
        for i in range(len(self.group_names)):
            print(f'Computation time {self.full_group_names[i]}: {self.time_means[self.group_names[i]]}.')
        if self.n_bandwidths != 1:
            for i in range(len(self.group_names)):
                print(f'Statistic mean median {self.full_group_names[i]}: {self.statistic_means[self.group_names[i]][self.bw[len(self.bw)-1]]}.')
                print(f'Threshold mean median {self.full_group_names[i]}: {self.threshold_means[self.group_names[i]][self.bw[len(self.bw)-1]]}.')
            
    def save(self):
        pickle.dump(self, open(self.fname, 'wb'))
        
    def get_lists(self, order=None, labels=None, wilson_intervals=False, z=1.96, median_rate=False):
        rejection_rates_list = []
        rejection_rates_upper_list = []
        rejection_rates_lower_list = []
        times_list = []
        
        if order is None:
            order = self.group_names
        if labels is None:
            labels = self.group_labels
        for name in order:
            if not median_rate:
                rejection_rates_list.append(self.rejection_rates[name])
            else:
                rejection_rates_list.append(self.rejection_rates_median[name])
            times_list.append(self.time_means[name])
            if wilson_intervals:
                if not median_rate:
                    hatp = self.rejection_rates[name]
                else:
                    hatp = self.rejection_rates_median[name]
                n_t = self.total_n_tests
                upper_bound = (hatp + z**2/(2*n_t))/(1+z**2/n_t) + z/(1+z**2/n_t)*np.sqrt(hatp*(1-hatp)/n_t+z**2/(4*n_t**2)) 
                lower_bound = (hatp + z**2/(2*n_t))/(1+z**2/n_t) - z/(1+z**2/n_t)*np.sqrt(hatp*(1-hatp)/n_t+z**2/(4*n_t**2)) 
                rejection_rates_upper_list.append(upper_bound)
                rejection_rates_lower_list.append(lower_bound)
            else:
                if not median_rate:
                    rejection_rates_upper_list.append(self.rejection_rates[name] + np.sqrt(self.rejection_rates[name]*(1-self.rejection_rates[name])/self.total_n_tests))
                    rejection_rates_lower_list.append(np.maximum(0,self.rejection_rates[name] - np.sqrt(self.rejection_rates[name]*(1-self.rejection_rates[name])/self.total_n_tests)))
                else:
                    rejection_rates_upper_list.append(self.rejection_rates_median[name] + np.sqrt(self.rejection_rates_median[name]*(1-self.rejection_rates_median[name])/self.total_n_tests))
                    rejection_rates_lower_list.append(np.maximum(0,self.rejection_rates_median[name] - np.sqrt(self.rejection_rates_median[name]*(1-self.rejection_rates_median[name])/self.total_n_tests)))
            
        return rejection_rates_list, rejection_rates_upper_list, rejection_rates_lower_list, times_list, labels
        
"""
%%%%%%%%%%%%%%%%%%%%%%% Functions to store and retrieve files %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""        
        
def format_int_list(mylist):
    formatted_list = ""
    for i in range(len(mylist)-1):
        formatted_list += str(mylist[i]) + '_'
    formatted_list += str(mylist[len(mylist)-1])
    return formatted_list            
            
def get_group_directories(args, test_groups, aggregated=False):
    resdir = dict()
    
    if args.name == 'gaussians':
        name_arguments = str(args.mean_diff)
    elif args.name == 'blobs':
        name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
    elif args.name == 'MNIST' or args.name == 'EMNIST': 
        name_arguments = str(args.p_even)
    elif args.name == 'Higgs': 
        if args.mixing:
            name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
        else:
            name_arguments = str(args.mixing)+'_'+str(args.null)
    elif args.name == 'sine':
        name_arguments = str(args.omega)
    else:
        name_arguments = 'None'
        
    for group in test_groups:
        #define directory to store results for estimators in group, and create it if needed 
        if aggregated:
            resdir[group] = os.path.join('res', args.name+'_'+group+'_'+name_arguments+'_aggregated')
        else:
            resdir[group] = os.path.join('res', args.name+'_'+group+'_'+name_arguments)
        if not os.path.exists(resdir[group]):
            os.makedirs(resdir[group], exist_ok=True)
            
    return resdir
    
def get_test_file_names(args, test_groups, resdir, n_tests=1, save=True, aggregated=False):
    #save = True to save files (in test.py), save = False to get files loaded (in postprocessing.py)
    
    formatted_wb_incomplete_list_multiples = format_int_list(args.wb_incomplete_list_multiples)
    formatted_block_g_list = format_int_list(args.block_g_list)
    formatted_n_features_list = format_int_list(args.n_features_list)
    formatted_block_g_n_features_list = format_int_list(args.block_g_list_ctt_rff+args.n_features_list_ctt_rff)
    if not aggregated:
        formatted_wb_block_size_list = format_int_list(args.wb_block_size_list)
        formatted_n_var = format_int_list(args.n_var)
        formatted_asymptotic_block_size_list = format_int_list(args.asymptotic_block_size_list)
        formatted_asymptotic_incomplete_list_multiples = format_int_list(args.asymptotic_incomplete_list_multiples)
        
    if args.name == 'gaussians':
        name_arguments = str(args.mean_diff)
    elif args.name == 'blobs':
        name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
    elif args.name == 'MNIST' or args.name == 'EMNIST': 
        name_arguments = str(args.p_even)
    elif args.name == 'Higgs': 
        if args.mixing:
            name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
        else:
            name_arguments = str(args.mixing)+'_'+str(args.null)
    elif args.name == 'sine':
        name_arguments = str(args.omega)
    else:
        name_arguments = 'None'
        
    group_arguments = dict()
    group_arguments['incomplete_wb'] = formatted_wb_incomplete_list_multiples
    group_arguments['ctt'] = formatted_block_g_list+'_'+str(args.different_compression)
    group_arguments['rff'] = formatted_n_features_list
    group_arguments['ctt_rff'] = formatted_block_g_n_features_list+'_'+str(args.s_rff)+'_'+str(args.s_permute)
    if not aggregated:
        group_arguments['block_wb'] = formatted_wb_block_size_list
        group_arguments['block_asymp'] = formatted_n_var+'_'+formatted_asymptotic_block_size_list
        group_arguments['incomplete_asymp'] = formatted_n_var+'_'+formatted_asymptotic_incomplete_list_multiples
        group_arguments['ctt'] = formatted_block_g_list+'_'+str(args.s_permute)
    
    if save:
        
        groupname = [None]*n_tests
        testname = [None]*n_tests

        for i in range(n_tests):
            groupname[i] = dict()
            testname[i] = dict()

        for i in range(n_tests):
            
            seed = str(args.seed)+'_'+str(i)
            
            for group in test_groups:
                if aggregated:
                    groupname[i][group] = args.name+'_'+group+'_aggregated_'+str(args.n_bandwidths)+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.B_2) +'_'+str(args.B_3)+'_' +seed+'_'+str(args.alpha)+'_'+name_arguments #+'_'+group_arguments[group]
                    
                    groupname_dir = os.path.join(resdir[group], groupname[i][group])
                    if not os.path.exists(groupname_dir):
                        os.makedirs(groupname_dir, exist_ok=True)
                    
                    testname[i][group] = dict()
                    for tname in args.estimators[group]:
                        testname[i][group][tname] = os.path.join(groupname[i][group],tname) 
                    groupname[i][group] += '_'+group_arguments[group]
                else:
                    groupname[i][group] = args.name+'_'+group+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+seed +'_'+str(args.alpha)+'_'+name_arguments #+'_'+group_arguments[group]
                    
                    groupname_dir = os.path.join(resdir[group], groupname[i][group])
                    if not os.path.exists(groupname_dir):
                        os.makedirs(groupname_dir, exist_ok=True)
                    
                    testname[i][group] = dict()
                    for tname in args.estimators[group]:
                        testname[i][group][tname] = os.path.join(groupname[i][group],tname)
                    groupname[i][group] += '_'+group_arguments[group]

        return groupname, testname
    
    else:
        
        groupname = dict()
        testname = dict()
            
        seed = '*'

        for group in test_groups:
            if aggregated:
                groupname[group] = args.name+'_'+group+'_aggregated_'+str(args.n_bandwidths)+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+seed +'_'+str(args.alpha)+'_'+name_arguments #+'_'+group_arguments[group]
                testname[group] = dict()
                for tname in args.estimators[group]:
                    testname[group][tname] = os.path.join(groupname[group],tname)
                groupname[group] += '_'+group_arguments[group]
            else:
                groupname[group] = args.name+'_'+group+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+seed +'_'+str(args.alpha)+'_'+name_arguments #+'_'+group_arguments[group]
                testname[group] = dict()
                for tname in args.estimators[group]:
                    testname[group][tname] = os.path.join(groupname[group],tname)
                groupname[group] += '_'+group_arguments[group]

        return groupname, testname
    

def get_fname_and_file_exists(args,test_groups,resdir,groupname,testname,save=True):
    
    # check if group names exist too
    
    if save:
        n_tests = len(testname)
        
        fname = [None]*n_tests
        file_exists = [None]*n_tests
        
        fname_group = [None]*n_tests
        file_exists_group = [None]*n_tests
        
        for i in range(n_tests):
            fname[i] = dict()
            file_exists[i] = dict()
            fname_group[i] = dict()
            file_exists_group[i] = dict()

        for i in range(n_tests):
            for group in test_groups:
                fname[i][group] = dict()
                file_exists[i][group] = dict()
                for tname in args.estimators[group]:
                    fname[i][group][tname] = os.path.join(resdir[group],testname[i][group][tname])
                    if os.path.exists(fname[i][group][tname]) and not args.interactive:
                        file_exists[i][group][tname] = True
                    else:
                        file_exists[i][group][tname] = False
                
                fname_group[i][group] = os.path.join(resdir[group],groupname[i][group])
                if os.path.exists(fname_group[i][group]) and not args.interactive:
                    file_exists_group[i][group] = True
                else:
                    file_exists_group[i][group] = False

        return fname, file_exists, fname_group, file_exists_group
    
    else:    
        fname = dict()
        file_exists = dict()
        fname_group = dict()
        file_exists_group = dict()

        for group in test_groups:
            fname[group] = dict()
            file_exists[group] = dict()
            for tname in args.estimators[group]:
                fname[group][tname] = os.path.join(resdir[group],testname[group][tname])
                if os.path.exists(fname[group][tname]) and not args.interactive:
                    file_exists[group][tname] = True
                else:
                    file_exists[group][tname] = False
                    
            fname_group[group] = os.path.join(resdir[group],groupname[group])
            if os.path.exists(fname_group[group]) and not args.interactive:
                file_exists_group[group] = True
            else:
                file_exists_group[group] = False

        return fname, file_exists, fname_group, file_exists_group
                      
def get_joint_group_directories(args, test_groups, aggregated=False):
    joint_resdir = dict()
    
    if args.name == 'gaussians':
        name_arguments = str(args.mean_diff)
    elif args.name == 'blobs':
        name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
    elif args.name == 'MNIST' or args.name == 'EMNIST': 
        name_arguments = str(args.p_even)
    elif args.name == 'Higgs': 
        if args.mixing:
            name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
        else:
            name_arguments = str(args.mixing)+'_'+str(args.null)
    elif args.name == 'sine':
        name_arguments = str(args.omega)
    else:
        name_arguments = 'None'
        
    for group in test_groups:
        # Define directory to store results for estimators in group, and create it if needed
        if aggregated:
            joint_resdir[group] = os.path.join('res_joint', args.name+'_'+group+'_'+name_arguments+'_joint_aggregated')
        else:
            joint_resdir[group] = os.path.join('res_joint', args.name+'_'+group+'_'+name_arguments+'_joint')
        if not os.path.exists(joint_resdir[group]):
            os.makedirs(joint_resdir[group], exist_ok=True)
            
    return joint_resdir
                      
def get_joint_filename(args, test_groups, aggregated=False):
    
    joint_filename = dict() 
    
    formatted_wb_incomplete_list_multiples = format_int_list(args.wb_incomplete_list_multiples)
    formatted_block_g_list = format_int_list(args.block_g_list)
    formatted_n_features_list = format_int_list(args.n_features_list)
    formatted_block_g_n_features_list = format_int_list(args.block_g_list_ctt_rff+args.n_features_list_ctt_rff)
    if not aggregated:
        formatted_wb_block_size_list = format_int_list(args.wb_block_size_list)
        formatted_n_var = format_int_list(args.n_var)
        formatted_asymptotic_block_size_list = format_int_list(args.asymptotic_block_size_list)
        formatted_asymptotic_incomplete_list_multiples = format_int_list(args.asymptotic_incomplete_list_multiples)
    
    if args.name == 'gaussians':
        name_arguments = str(args.mean_diff)
    elif args.name == 'blobs':
        name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
    elif args.name == 'MNIST' or args.name == 'EMNIST': 
        name_arguments = str(args.p_even)
    elif args.name == 'Higgs': 
        if args.mixing:
            name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
        else:
            name_arguments = str(args.mixing)+'_'+str(args.null)
    elif args.name == 'sine':
        name_arguments = str(args.omega)
    else:
        name_arguments = 'None'
        
    group_arguments = dict()
    group_arguments['incomplete_wb'] = formatted_wb_incomplete_list_multiples
    group_arguments['ctt'] = formatted_block_g_list+'_'+str(args.different_compression)+'_'+str(args.s_permute)
    group_arguments['rff'] = formatted_n_features_list
    group_arguments['ctt_rff'] = formatted_block_g_n_features_list+'_'+str(args.s_rff)+'_'+str(args.s_permute)
    if not aggregated:
        group_arguments['block_wb'] = formatted_wb_block_size_list
        group_arguments['block_asymp'] = formatted_n_var+'_'+formatted_asymptotic_block_size_list
        group_arguments['incomplete_asymp'] = formatted_n_var+'_'+formatted_asymptotic_incomplete_list_multiples
        group_arguments['ctt'] = formatted_block_g_list
    
    for group in test_groups:
        if aggregated:
            joint_filename[group] = args.name+'_'+group+'_aggregated_'+str(args.n_bandwidths)+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.B_2) +'_'+str(args.B_3)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+name_arguments+'_'+group_arguments[group]
        else:
            joint_filename[group] = args.name+'_'+group+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+name_arguments+'_'+group_arguments[group]
        
    return joint_filename
                      
def get_fname_joint(args,test_groups,joint_resdir,joint_filename):
    
    joint_fname = dict()
    
    for group in test_groups:
        joint_fname[group] = os.path.join(joint_resdir[group],joint_filename[group])
            
    return joint_fname
       