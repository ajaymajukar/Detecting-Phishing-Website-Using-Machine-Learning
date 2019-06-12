import pandas as pd
import feature_extraction as fe

urls = open("urls.txt", 'r')
new_dataset = open("new_dataset.csv", 'a')

features = []

for url in urls.readlines():
    
    label = int(url.strip().split(',')[1])
    feat = fe.generate_data_set(url.split(',')[0])
    feat += [label]
    print(str(feat))
    f = str(feat)[1:-1]
    new_dataset.write(f)
    new_dataset.write("\n")
    #features.append(feat)
new_dataset.close()
urls.close()
#features_df = pd.DataFrame(features)
#features_df.to_csv("new_dataset.csv")
#print(features)



