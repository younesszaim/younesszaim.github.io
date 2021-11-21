## Practical Statistics for Data Scientist

Two goals underlie these notes :

• To lay out, in digestible, navigable, and easily referenced form, key concepts from
statistics that are relevant to data science.

• To explain which concepts are important and useful from a data science perspective,
which are less so, and why.

---

### Exploratory Data Analysis
There are two basic types of structured data: numeric and categorical.

• Numeric data : Data that are expressed on a numeric scale : 

      - Continuous : Data that can take on any value in an interval. (Synonyms: interval, float, numeric)
      - Discrete : Data that can take on only integer values, such as counts. (Synonyms: integer, count)
      
• Categorical : Data that can take on only a specific set of values representing a set of possible categories. (Synonyms: enums, enumerated, factors, nominal) 

      - Binary : A special case of categorical data with just two categories of values, e.g., 0/1, true/false. (Synonyms: dichotomous, logical, indicator, boolean)
      - Ordinal : Categorical data that has an explicit ordering. (Synonym: ordered factor)
      
The typical frame of reference for an analysis in data science is a rectangular data object, like a spreadsheet or database table :

    - Data frame : Rectangular data (like a spreadsheet) is the basic data structure for statistical and machine learning models.
    - Feature : A column within a table is commonly referred to as a feature. (Synonyms: attribute, input, predictor, variable)
    - Outcome : Synonyms dependent variable, response, target, output
    - Records : A row within a table is commonly referred to as a record. (Synonyms case, example, instance, observation, pattern, sample)
 
 
#### Estimates of Location : Variables with measured or count data might have thousands of distinct values. A basic step in exploring your data is getting a “typical value” for each feature

    - Mean : The sum of all values divided by the number of values. (Synonym average)
    
```python
Dataset.Column.mean()
```
    
    - Weighted mean : The sum of all values times a weight divided by the sum of the weights. (Synonym weighted average)
    
```python
import numpy as np
Weighted_Mean = np.average(Dataset.Column, weights=Dataset.ColumnWeight)
```
    
    - Median : The value such that one-half of the data lies above and below. (Synonym 50th percentile)
    
```python
Dataset.Column.median()
```
    - Percentile The value such that P percent of the data lies below. (Synonym quantile)
    
```python
import numpy as np
Quantile_at_P = np.quantile(Dataset.Column, q=Percentile)
````
    - Weighted median The value such that one-half of the sum of the weights lies above and below the sorted data.
    
```python
import wquantiles
Weighted_median = wquantiles.median(Dataset.Column, weights=Dataset.ColumnWeight)
````
    - Trimmed mean The average of all values after dropping a fixed number of extreme values. (Synonym truncated mean)
    
```python
from scipy.stats import trim_mean
Trim_mean = trim_mean(Dataset.Column, proportiontocut)
````

#### Estimates of Variability : Location is just one dimension in summarizing a feature. A second dimension, variability,also referred to as dispersion, measures whether the data values are tightly clustered or spread out.

    - Deviations : The difference between the observed values and the estimate of location. (Synonyms errors, residuals)
    - Variance : TThe sum of squared deviations from the mean divided by n – 1 where n is the number of data values. (Synonym mean-squared-error)
    - Standard deviation : The square root of the variance.
    
```python
Dataset.Column.std()
````  

    - Mean absolute deviation : The mean of the absolute values of the deviations from the mean. (Synonyms l1-norm, Manhattan norm)
    - Median absolute deviation from the median : The median of the absolute values of the deviations from the median.
    
```python
from statsmodels import robust
robust.scale.mad(Dataset.Column)
````  

    - Range : The difference between the largest and the smallest value in a data set.
    - Order statistics Metrics : based on the data values sorted from smallest to biggest. (Synonym ranks)
    - Percentile : The value such that P percent of the values take on this value or less and (100–P) percent take on this value or more. (Synonym quantile)
    - Interquartile range : The difference between the 75th percentile and the 25th percentile. (Synonym IQR)
    
```python
Dataset.Column.quantile(0.75) - Dataset.Column.quantile(0.25)
````      

#### Exploring the Data Distribution : Each of the estimates we’ve covered sums up the data in a single number to describe the location or variability of the data. It is also useful to explore how the data is distributed overall.
    
    - Boxplot : A plot introduced by Tukey as a quick way to visualize the distribution of data. (Synonym box and whiskers plot)
    - Frequency table : A tally of the count of numeric data values that fall into a set of intervals (bins).
    - Histogram : A plot of the frequency table with the bins on the x-axis and the count (or proportion) on the y-axis. While visually similar, bar charts should                  not be confused with histograms.
    - Density plot : A smoothed version of the histogram, often based on a kernel density estimate.

    

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
