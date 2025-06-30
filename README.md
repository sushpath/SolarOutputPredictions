### Overview
I was curious about how the weather affects solar power generation for my solar panels. And coincidentally, this could be a nice exercise for learning ML algorithms. 

1. Weather data : For the weather data, following sources seemed relevant
  - [IMD wrapper](https://github.com/rtdtwo/india-weather-rest)
  - [Indian API](https://indianapi.in/weather-api)
  - [Visual Crossing](https://www.visualcrossing.com/weather-query-builder/)
Among these, only Visaul Crossing Query builder provided historial data from January 2025 with detailed temperature, cloud cover and precipitation features
2.  Solar ouput data : The iSolarCloud app allows exporting generation data in CSV.

### Data Preprocessing
- Weather data had some columns like `feelslike temp` which were cleaned out.
- A few outliers had to be removed such as, power cuts causing the solar plant's grid export to be 0.
- Data was normalized with zscore as a few columns ranged in 100s while a few columsn were in fractions.

### Feature engineering
- `sunset` and `sunrise` times were not directly useful for prediction. However, `solartime` = `sunset` - `sunrise` meant how long the sun was up, which definitely contributed to the solar generation.
- As the input features 12 usually indicated high bias (underfitting), `PolynomialFeatures` (degree=2) with interactions were added to the dataset. This always showed significant improvement in accuracy scores.
 
### ML algorithms
Trying out various algorithms and fine-tuing them turned out to be interesting, bit cumbersome but definitely rewarding.
- Accuracy was computed using R2 scores of both training and test data
- Best accuracy score below is for the test data when training R2 score is > 80% but not 100% 

| Algorithm Type | Variations | Best accuracy (R2 score) |
|----------------|------------|--------------------------|
| Linear Regression | Linear, SGD, Ridge, LassoCV | Ridge(L2) = 48% and LassoCV = 47.7% |
| Neural Network    | 2 vs 3 hidden layers, w/ and w/o `dropout`, data shuffing | With 3 layers + repeated training with data shuffling = 77% | 
| Decision Trees    | `max_depth`, `n_leaf_nodes`, `max_features` | Limiting the depth, feature count avoided overfitting with score 59% |
| Random Forests    | `max_features`, `n_estimators` | Limiting the estimators and features limited the overfitting with score 53% |

### Learnings
1. Linear regression algos exhibit limitations as feature count and variance increases. The problem is typically high bias(underfitting) which can be partially overcome with `PolynomialFeatures`
2. Neuaral network are unpredicatable for smaller datatsets. However, with repeated training and higher data variety, the accuracy can be improved.
3. Decision Trees and Random Forests have tendancy of high variance(overfitting) on training data. With hyper-parameter tuning the test data accuracy can be improved while making sure the model can generalize well.
