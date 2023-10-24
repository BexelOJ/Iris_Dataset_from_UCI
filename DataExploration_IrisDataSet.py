#!/usr/bin/env python
# coding: utf-8

# # Student Name: Bexel O J
# 
# # Bits ID : 2022MT93048
# 
# ## Course Name: SEZG568 Applied Machine Learning
# 
# ## Probelm_Statement: Data Exploration using Iris Dataset

# In[ ]:





# ## Scatter Plots : Sepal Length vs Petal Length

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset("iris")

# Create a scatter plot for Sepal Length vs Petal Length
sns.scatterplot(x="sepal_length", y="petal_length", data=iris)

# Set labels and title
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")

# Show the plot
plt.show()


# In[ ]:





# ## Scatter Plots : Petal Width vs Sepal Width

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset("iris")

# Create a scatter plot for Petal Width vs Sepal Width
sns.scatterplot(x="petal_width", y="sepal_width", data=iris)

# Set labels and title
plt.xlabel("Petal Width (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Petal Width vs Sepal Width")

# Show the plot
plt.show()


# In[ ]:





# ## Scatter Plots : Petal Length vs Sepal Length

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset("iris")

# Create a scatter plot for Petal Length vs Sepal Length
sns.scatterplot(x="petal_length", y="sepal_length", data=iris)

# Set labels and title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Sepal Length (cm)")
plt.title("Petal Length vs Sepal Length")

# Show the plot
plt.show()


# In[ ]:





# ## For all numeric attributes, compute Mean and Standard Deviation

# In[4]:


import pandas as pd

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None,
                  names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Calculate the mean and standard deviation for all numeric attributes
numeric_attributes = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
mean_values = numeric_attributes.mean()
std_deviation = numeric_attributes.std()

print("Mean Values:")
print(mean_values)

print("\nStandard Deviation:")
print(std_deviation)


# In[ ]:





# ## Create 5-point summary for all numeric attributes

# In[5]:


import pandas as pd

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None,
                  names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Select numeric attributes
numeric_attributes = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

# Calculate the five-number summary using the describe() function
five_number_summary = numeric_attributes.describe().loc[['min', '25%', '50%', '75%', 'max']]

print("Five-Number Summary:")
print(five_number_summary)


# ## Visualization

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None,
                  names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Select numeric attributes
numeric_attributes = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

# Create a box plot
plt.figure(figsize=(10, 6))
numeric_attributes.boxplot(vert=False)
plt.title("Five-Number Summary for Numeric Attributes")
plt.xlabel("Value")
plt.ylabel("Attributes")
plt.show()


# In[ ]:





# ## Create histogram (with 5 bins) from Sepal Length and Petal Width

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None,
                  names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Select Sepal Length and Petal Width
sepal_length = iris["sepal_length"]
petal_width = iris["petal_width"]

# Create histograms with 5 bins
plt.figure(figsize=(10, 6))

plt.hist(sepal_length, bins=5, alpha=0.5, label="Sepal Length")
plt.hist(petal_width, bins=5, alpha=0.5, label="Petal Width")

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histograms for Sepal Length and Petal Width")
plt.legend()

plt.show()


# In[ ]:




