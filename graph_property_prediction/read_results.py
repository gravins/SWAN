import os
import pandas

path = './gpp_exp/GraphPropPred/'

final_df = {'diam':{}, 'sssp':{}, 'ecc':{}}
for t in ['diam', 'sssp', 'ecc']:
    tmp_path = os.path.join(path,t) # ./gpp_exp/..../diam 
    for pp in os.listdir(tmp_path):
		
        exp_path = os.path.join(tmp_path,pp) # ./gpp_exp/..../diam/SONAR
        print(exp_path)

        if not os.path.exists(os.path.join(exp_path, 'results.csv')):
            print('results.csv is not present')
            continue
		
        df = pandas.read_csv(os.path.join(exp_path, 'results.csv'))

        row = df.sort_values('avg best_val_score').iloc[0]
        avg, std = row['avg best_test_score'], row['std best_test_score']
        name = pp
        val = str(round(avg,4)) + '$_{\pm' + str(round(std, 4)) + '}$'
        val_no_round = str(avg) + '$_{\pm' + str(std) + '}$'
        final_df[t][name] = val

final_df = pandas.DataFrame(final_df, columns=final_df.keys())
final_df = final_df.sort_index()
print(final_df)
