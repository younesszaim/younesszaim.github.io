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
 
 
 
 
 
 
 
 
