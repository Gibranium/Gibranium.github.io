# Position Cluster Project
This was a project I built with the intent of creating my own position feature for players based on where on the pitch they actually performed their actions and other features: relative frequency of shots, defensive actions and so on.

- [Jupyter Notebook](https://gibranium.github.io/positioncluster/CLUSTERING-POSITIONS.html)
- Examples:
  
![Unknown-1](https://github.com/user-attachments/assets/014dcbe8-e557-423a-a503-9c138d39625a)

![Unknown-2](https://github.com/user-attachments/assets/e0f884a9-ea58-415d-8890-5ee4940c0757)

![Unknown](https://github.com/user-attachments/assets/25d6a967-88bc-4c29-9025-3ecb183d2f8d)

# 2.0 Version
This is divided in 2 notebooks, one to create the necessary dataframe for each season and the other to do the clustering operation.
This works directly on the data created by the heatmap function that I use almost everywhere I need it, therefore is way more precise in clustering together players and doesn't need metrics of quality like xG, xA. Still is time intensive as the mapping of clusters id and positions is done manually. 

- [Jupyter Notebook for the dataframes preparation](https://gibranium.github.io/positioncluster/CLUSTERING-POSITIONS-CREATION.html)
- [Jupyter Notebook for the actual clustering](https://gibranium.github.io/positioncluster/CLUSTERING-POSITIONS_CLUSTERING.html)
- Examples:
  
![Unknown-1](https://github.com/user-attachments/assets/159fd80f-41df-4327-84c9-ccd61db10959)

![Unknown](https://github.com/user-attachments/assets/a6984e61-0b91-4ab1-95bb-3bc0747a519f)
