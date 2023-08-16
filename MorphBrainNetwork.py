import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import math
import yaml

from sklearn.linear_model  import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

import argparse


#Source: https://www.youtube.com/watch?v=fykGWCplQIc&list=PLug43ldmRSo0bX8cOSuMWinXs-xem9q4o&index=38&ab_channel=BASIRALab


def calculate_max_principal_curvature(mean_curv, gaus_curv):
# Ref : https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.25776
# Cortical thickness systematically varies with curvature and depth in healthy human brains
# Nagehan Demirci, Maria A. Holland
# if the mean_curv**2 - gaus_curv < 0, then just have mean curvature.
    max_principal_curvature = mean_curv
    
    if (mean_curv**2 - gaus_curv) >= 0:
        max_principal_curvature = mean_curv + math.sqrt (mean_curv**2 - gaus_curv)   
    
    return max_principal_curvature

def calculate_min_principal_curvature(mean_curv, gaus_curv):
    min_principal_curvature = mean_curv - math.sqrt (mean_curv**2 - gaus_curv)
    return min_principal_curvature


def low_order_network(sub_path,subjects):

	df = pd.DataFrame()
	for subject in subjects:
		#for every hemisphere    
		for hem in hems:
			# Specify the file path
			file_path = hem+'.aparc.DKTatlas.stats'    
			column_names = []   
			#print(subject)
			try:
				with open(os.path.join(sub_path,subject,'stats', file_path), 'r') as file:
					lines = file.readlines()
					last_comment_line = [line.strip() for line in lines if line.startswith('#')][-1]
					column_names = last_comment_line.split()[2:]  # Exclude the first two elements: '#' and 'ColHeaders'

				# Read the text file into a DataFrame, skipping the specified number of lines
				#delimiter parameter is set to '\s+', which uses a regular expression to handle one or more consecutive whitespaces as the delimiter        
				df = pd.read_csv(os.path.join(sub_path,subject,'stats', file_path), comment='#', delimiter='\s+', skiprows=1, names=column_names)
				#display(df)
				#n_ROIS = df.shape[0]
				
				#The principal curvatures can be computed from the mean curvature (H) and Gaussian curvature (K) using the following equations:
				#K = H^2 - K
				df['MaxPrincipalCurv'] = df.apply(lambda row: calculate_max_principal_curvature(row['MeanCurv'], row['GausCurv']), axis=1)
				
				#high order matrix for ?hemisphere
				lo_mtx =  np.zeros((df.shape[0],df.shape[0],len(frontal_views)))
				
				#for every frontal view
				for fv in range(lo_mtx.shape[2]):              
					for i in range(n_ROIS):
						for j in range(n_ROIS):
							#compute the absolute difference between ROI_i and ROI_j 
							ROI_i_mean = df.loc[i,frontal_views[fv]]
							ROI_j_mean = df.loc[j,frontal_views[fv]]
							lo_mtx_ij = np.abs(ROI_i_mean - ROI_j_mean)
							#print(i , j , ho_lh_ij)
							lo_mtx[i,j,fv]  = lo_mtx_ij
					#print(subject, ho_lh.shape)
					#save subject LO matrix in directory
				np.save(os.path.join(sub_path, subject, f'lo_{hem}.npy'), lo_mtx)
				visualize(lo_mtx, sub_path, subject, hem, frontal_views)
				print("File and matrix saved ", os.path.join(sub_path, subject, f'lo_{hem}.npy'))

			except Exception as e:
				print("Exception caught: ", e)


def high_order_network(sub_path,subjects):
	#ho_mtx = np.zeros((n_ROIS**2,n_ROIS**2))
	for subject in subjects:	   
		ho_mtx = np.zeros((n_ROIS**2,n_ROIS**2))
		#for every hemisphere
		for hem in hems:
			#load the npy file	        
			try:
				lo_mtx = np.load(os.path.join(sub_path, subject, f'lo_{hem}.npy'))	             
				for i in range(n_ROIS):
					for j in range(n_ROIS):	                    
						#Pair 2
						for p in range(n_ROIS):
							for q in range(n_ROIS): 
								
								# Avoid correlating pairs of ROIs with themselves
								if ((i != p) or (j != q)):         

									#compute correlation between these two pairs of ROI
									#Get the vector Yij across all views
									row_pair1 = lo_mtx[i,j,:].flatten()
									row_pair2 = lo_mtx[p,q,:].flatten()                            

									if np.any(row_pair1) and np.any(row_pair2):
										#print(f'Pair ({i} {j}) x Pair ({p} {q})')
										#print(row_pair1,row_pair2)
										# Compute the correlation between the pairs of ROIs across the 4 dimensions
										correlation = np.corrcoef(row_pair1, row_pair2)[0, 1]
										#print(correlation)
										# Store the correlation in the 2D matrix
										ho_mtx[i*n_ROIS+j, p*n_ROIS+q] = correlation
				np.save(os.path.join(sub_path, subject, f'ho_{hem}.npy'), ho_mtx)
				print("File saved!", ho_mtx.shape)      
			except Exception as e:
				print("Exception occurred ", e)


def feature_extraction(sub_path,subjects):
	#every  subject / hem has a feature vector
	#end up with n_sub x n_feat matrix, per hemisphere 

	for hem in hems:    
		print(hem)    
		conn_feat_mtx = []   
		data=dict()
		#data ["true_labels"] = []
		data ["ID"] = []
		data ["Age"] = []
		data ["Dataset"] = []
		for index, row in subjects.iterrows(): 
			#do feature extraction on a set number of subjects to get the matrix of the included ones
            #modified this for controls to take into account the group name, otherwise it's an iteration through subjects.
            #for every hemisphere 
			subid = row['ID']
			age = row['AGE']
			dataset= row['Dataset']
			#load the npy file
			matrix_path = os.path.join(sub_path, subid)

			try:
				lo_exists = os.path.exists(os.path.join(matrix_path, f'lo_{hem}.npy'))
				ho_exists = os.path.exists(os.path.join(matrix_path, f'ho_{hem}.npy'))

				if lo_exists and ho_exists:                
					ho_mtx = np.load(os.path.join(matrix_path, f'ho_{hem}.npy'))
					row_indices, col_indices = np.triu_indices(ho_mtx.shape[0], k=1)  # k=1 excludes the diagonal
					# Ravel the upper triangle elements
					feature_vector_s = ho_mtx[row_indices, col_indices]
					lo_mtx = np.load(os.path.join(matrix_path, f'lo_{hem}.npy'))
					feature_vector_s = np.concatenate((feature_vector_s, np.ravel(lo_mtx)),axis=0)  

					if (len(feature_vector_s) == int(4*n_ROIS**2 + (ho_mtx.shape[0] * (ho_mtx.shape[0] -1)/2))): #4*31² + (961 * 961 - 961) / 2 = 961 + 461280
						conn_feat_mtx.append(feature_vector_s.reshape(-1,1))
						data["ID"].append(subid)
						data["Age"].append(age)
						data['Dataset'].append(dataset)
							
					else:
						print("Warning: ", subid, len(feature_vector_s))

			except Exception as e:
				print("Exception caught: ", e)
		feature_matrix = np.squeeze(np.array(conn_feat_mtx))
		print(f'Final matrix shape {feature_matrix.shape}')
		np.save(os.path.join(sub_path,f'conn_feat_mtx_{hem}.npy'), feature_matrix)
		#data['Age'] = [0] * len(data['ID'])
		df=pd.DataFrame.from_dict(data)
		df.to_csv(os.path.join(sub_path,f'conn_feat_annot.csv'),index=False)
	    

def get_labels(subid,hem,sub_path):
	subject=subid #test subject, could be anything
	ho_mtx = np.load(os.path.join(sub_path, subject, f'ho_{hem}.npy'))
	row_indices, col_indices = np.triu_indices(ho_mtx.shape[0], k=1)  # k=1 excludes the diagonal
	# Ravel the upper triangle elements
	feature_vector_s = ho_mtx[row_indices, col_indices]
	print(len(feature_vector_s))

	n_ROIS = 31
	#### LO LABELS ####
	# Print the list of (pair x pair) combinations
	lo_pairs=[]
	for i, j in itertools.combinations(range(n_ROIS), 2):
		label_i = ROIS[i]
		label_j = ROIS[j]
		lo_pairs.append((label_i, label_j))

	print(len(lo_pairs))
	labels = pd.DataFrame.from_dict({"lo_pairs":lo_pairs})
	labels.to_csv(os.path.join(sub_path, 'lo_Labels.csv'), index=False)

	#### HO LABELS ####
	ho_pairs = []
	for i in range(n_ROIS):
		for j in range(n_ROIS):
			# Exclude pairs where i equals j to avoid duplicate entries
			#if i != j:
			ho_pairs.append(((i, j), (i, j)))

	print(len(ho_pairs))

	labels = pd.DataFrame.from_dict({"ho_pairs":ho_pairs})
	labels.to_csv(os.path.join(sub_path, 'ho_Labels.csv'), index=False)

	#### Averaged and Concatenated matrices labels ####
	# Create a 31x31 matrix for illustration purposes
	matrix_31x31 = np.arange(n_ROIS**2).reshape(n_ROIS, n_ROIS)

	# # Flatten the matrix
	# flattened_vector = matrix_31x31.flatten()

	# Get the row and column indices corresponding to the flattened vector
	row_indices, col_indices = np.unravel_index(np.arange(n_ROIS**2), (n_ROIS, n_ROIS))

	# Create a mapping between indices and labels
	index_label_mapping = [(row, col) for row, col in zip(row_indices, col_indices)]
	labels = []
	# Print the labels for the flattened vector
	for index, label in enumerate(index_label_mapping):
		labels.append((ROIS[label[0]], ROIS[label[1]]))

	df_concatenated_labels = pd.DataFrame({"Label": labels * 4}) #for every frontal view
	df_averaged_labels  = pd.DataFrame({"Label": labels})

	df_concatenated_labels.to_csv(os.path.join(sub_path, 'df_concatenated_labels.csv'), index=False)
	df_averaged_labels.to_csv(os.path.join(sub_path, 'df_averaged_labels.csv'), index=False)

	# Get the indices of the upper triangle
	row_indices, col_indices = np.triu_indices(n_ROIS**2, k=1)  # k=1 excludes the diagonal

	# Create a list to store (i, j, p, q) pairs for the upper triangle
	upper_triangle_pairs = []

	# Map indices to (i, j, p, q) pairs
	for row, col in zip(row_indices, col_indices):
		i = row // n_ROIS
		j = row % n_ROIS
		p = col // n_ROIS
		q = col % n_ROIS
		upper_triangle_pairs.append((ROIS[i], ROIS[j], ROIS[p], ROIS[q]))

	print(len(upper_triangle_pairs))

	labels = pd.DataFrame({"Label": upper_triangle_pairs})

	df_concatenated = pd.read_csv(os.path.join(sub_path, 'df_concatenated_labels.csv'))
	df_ = pd.concat([labels,df_concatenated],axis=0)
	print(len(df_))
	print(df_.head)


def visualize(mtx, sub_path, subject, hem, frontal_views):
	
	#for index, row in controls.iterrows(): #do feature extraction on a set number of subjects to get the matrix of the included ones
	#modified this for controls to take into account the group name, otherwise it's an iteration through subjects.
	#for every hemisphere   	
	try:
		
		connectivity_matrix = mtx #np.load(os.path.join(main_dir,"Controls",group,'recon-all', subject, "lo_rh.npy"))
		print(connectivity_matrix.shape)
		# Calculate the number of subplots needed
		num_plots = connectivity_matrix.shape[2]	
		
		for fv in range(num_plots):
			
			fig, ax = plt.subplots(figsize=(8, 6))
			# Flatten the connectivity matrix
			weights = connectivity_matrix[:,:,fv]

			# Remove zero or NaN weights (optional)
			#weights = weights[np.nonzero(weights)]

			# Plot the histogram in the respective subplot
			sns.heatmap(weights, fmt=".1f",cmap="coolwarm", ax=ax)
			plt.title(f'Heatmap for {frontal_views[fv]} Subject {subject}')
			plt.tight_layout()
			plt.savefig(os.path.join(sub_path, subject, f'heatmap_{frontal_views[fv]}.png'))
			plt.close()
	except Exception as e:
		print("Exception occurred: ", e)

	# Adjust spacing between subplots
	plt.tight_layout()
	# Show the plot
	#plt.show()
	plt.close()


# Get the command-line arguments
parser = argparse.ArgumentParser(description='Help menu running MorphBrainNetwork')
parser.add_argument('--sub_path', '-sub_path', type=str, nargs='?',
					help='Path to subjects folders')

parser.add_argument('--sub_df', '-sub_df', type=str, nargs='?',
					help='Dataframe containing ID and Age columns')
parser.add_argument('--config_file','-config_file',type=str,nargs='?',help='Path to config file')
parser.add_argument('--lo','-lo',type=str,nargs='?',default="y", help='Lower order matrix y/n')
parser.add_argument('--ho','-ho',type=str,nargs='?',default="y", help='Higher order matrix y/n')
parser.add_argument('--fe','-fe',type=str,nargs='?',default="y", help='Feature extraction y/n')

args = parser.parse_args()
sub_path = args.sub_path
sub_df = args.sub_df
config_file = args.config_file
lo = args.lo
ho = args.ho
fe = args.fe

with open(config_file, 'r') as ymlfile:
	cfg = yaml.safe_load(ymlfile)

print(cfg)
n_ROIS=cfg['n_ROIS']
ROIS=cfg['ROIS']
hems=cfg['hems']
frontal_views=cfg['frontal_views']

def main():

	dirs = os.listdir(sub_path)
	subjects = [d for d in dirs if os.path.isdir(os.path.join(sub_path, d))]
	if "fsaverage" in subjects:
		subjects.remove("fsaverage")

	print(f'# subjects {len(subjects)}')
	print(f'NROIS: {n_ROIS} hems {hems} fv {frontal_views}')

	if lo=="y":
		print('Performing lower order matrix construction...')
		low_order_network(sub_path,subjects)

	if ho=="y":
		print('Performing higher order matrix construction...')
		high_order_network(sub_path,subjects)

	if fe=="y":
		print('Performing feature extraction...')
		subjects = pd.read_csv(sub_df)
		feature_extraction(sub_path,subjects)

	#uncomment if you need labels output in a CSV
	#get_labels(df.iloc[0]['ID'],hems[0]) #pass any subject and any hemisphere 
  

if __name__ == "__main__":
    main()

