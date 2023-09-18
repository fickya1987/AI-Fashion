<img src="data/AUEB.png" />  <img src="data/MSc_BA.png" />
___
# Athens University of Economics and Business
# School of Business
# Department of Management Science & Technology
# Master of Science in Business Analytics
___
<table style='float:left;font-size: 20px;'>
    <tr>
        <th style='text-align: left;'>Program:</th>
        <td style='text-align: left;'>Full-Time</td>
    </tr>
    <tr>
        <th style='text-align: left;'>Quarter:</th>
        <td style='text-align: left;'>3rd (Spring Quarter)</td>
    </tr>
    <tr>
        <th style='text-align: left;'>Course:</th>
        <td style='text-align: left;'>Machine Learning & Content Analytics</td>
    </tr>
    <tr>
        <th style='text-align: left;'>Assignment:</th>
        <td style='text-align: left;'>Image Classification for Outfit Suggestions</td>
    </tr> 
    <tr>
        <th style='text-align: left;'>Students (Registration No):</th>
        <td style='text-align: left;'>Yankee Team (f2822203, f2822215, f2822217, f2822218),</td>
    </tr>
</table>

# AI Fashion
A project for the Machine Learning and Content Analytics course of the AUEB's MSc in Business Analytics, made by the Yankee team of the Full-Timers.

 ## Project Description
Our main project idea, named **Image Classification for Outfit Suggestions** is to create an application, named **dressmeup**, that can create outfit combinations based on user’s input photos. So, the business question, that our project seeks a good solution, is: 
**What to wear today (or a specific given day) from the clothes I have?**, and 
**Does an article (garment) that I like can be well-combined with the clothes I have?**.

## Data Collection
The data source that our data mostly came from was [Fashion Product Images Dataset from Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset), which contains 44 thousands products with multiple category labels, descriptions and high-resolution images. Also, from [thomascamminady GitHub repository](https://github.com/thomascamminady/APDOCC/tree/master), we got the digital version of the ['A Dictionary of Color Combinations' by Sanzo Wada](https://www.amazon.com/Dictionary-Color-Combinations-Various/dp/4861522471) book’s collection of color combinations, in order to use for clothes combination.

## Dataset Overview
The dataset contains 44,446 products. Each one of the 44k products is identified by an ID. A map to all products exists in the “styles.csv” file, containing various features for each product. The features that we found most useful for our project and used were gender, subCategory, articleType, season, and usage. The images of 44,441 products can be fetched from ‘images/{ID}.jpg’. Gender attribute has 5 unique values, subCategory has 45, articleType has 143, season has 5, and usage has 9 unique values. The images are of resolution 1800x2400. In the color combination repository exist 348 combinations from 157 different colors.

## Data Processing Steps
1.	Filtered the dataset to keep only top and bottom-wear clothing items, removing other categories, but retained 'Shorts' from the 'Loungewear and Nightwear' subCategory.
2.	Kept only specific usage categories: casual, formal, sports, smart casual, party, and travel clothes.
3.	Excluded gender categories 'unisex', 'boys', and 'girls'; focused only on 'Men' and 'Women' clothing.
4.	Created a new column for image paths and removed rows without associated images.
5.	Imputed null values in the 'usage' and 'season' columns.
6.	Null 'usage' values in three shirt products set to 'Formal' and null 'season' value in a T-shirt product set to 'Summer'.
7.	Recategorized 'Smart Casual', 'Party', and 'Travel' clothes as 'Casual' due to low sample sizes.
8.	Updated 'Winter' and 'Spring' clothes to 'Fall/Winter' and 'Spring/Summer' respectively, based on main fashion week seasons.
9.	Augmented the dataset by applying image rotations, scaling, and horizontal flips to achieve 1,000 images per articleType.
10.	Encoded category labels alphabetically, assigning integer values (e.g., Casual=0, Formal=1, Sports=2).
11.	Saved the processed data as a CSV file for distribution among project members for parallel work.

## Methodology
In our project, we utilize a diverse set of algorithms to tackle various aspects of our tasks:

1. **Stochastic Gradient Descent (SGD) Classifier**:
   - Utilized as a Support Vector Machine (SVM) classifier.
   - Implements regularized linear models with stochastic gradient descent.
   - Supports hinge loss and L2 penalty.
   - Allows model updates with a decreasing strength schedule (learning rate).

2. **Passive Aggressive Classifier**:
   - Belongs to a family of algorithms designed for large-scale learning.
   - Uses hinge loss for classification.
   - Incorporates a regularization parameter, C, to control model behavior.

3. **Convolutional Neural Network (CNN) with ResNet50 Transfer Learning**:
   - Leverages CNNs for image classification.
   - Implements transfer learning from the ResNet50 architecture.
   - ResNet50 is a 50-layer CNN known for its deep residual connections.
   - These residual networks are designed with stacked residual blocks.

4. **Convolutional Neural Network (CNN) with EfficientNetV2-S Transfer Learning**:
   - Adopts another powerful CNN architecture, EfficientNetV2S.
   - EfficientNet models are created using Neural Architecture Search (NAS).
   - EfficientNetV2S is a compact model with faster training and better parameter efficiency.
   - It captures intricate image features efficiently, enhancing classification accuracy.

5. **K-Means Clustering**:
   - Applied to cluster pixels within specified image regions.
   - Extracts the most dominant color from the clusters, i.e., the color with the highest pixel count.

These algorithms collectively empower our project to handle image classification, feature extraction, and color analysis tasks effectively. Each algorithm plays a specific role, contributing to the overall success of our fashion item classification model.

## Tools
Python Programming Language.

Scikit-Learn and Tensorflow libraries.

Streamlit open-source app framework to create beautiful web app.

SQLite Database Engine.
