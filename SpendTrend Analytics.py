import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('D:\ML\Mall_Customers.csv')
X=df.iloc[:,[3,4]].values

wcss=[]

from sklearn.cluster import KMeans
for j in range(1,11):
    print(j)
    kmeans_obj=KMeans(n_clusters=j,random_state=42,init="k-means++")
    kmeans_obj.fit(X)
    wcss.append(kmeans_obj.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow method: Finding optimal Clusters")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()

kmeans_obj=KMeans(n_clusters=5,init="k-means++",random_state=42)
y_pred=kmeans_obj.fit_predict(X)

colors=["red","blue","green","orange","brown"]
for i in range(0,5):
    plt.scatter(X[y_pred == i,0],X[y_pred == i,1],c=colors[i],s=100,label=('Cluster ' + str(i+1)))
plt.scatter(kmeans_obj.cluster_centers_[:,0],kmeans_obj.cluster_centers_[:,1],s=70,c="black",label='Centroids')
plt.title("Clusters of customers")
plt.xlabel("Annual Income (1000$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
