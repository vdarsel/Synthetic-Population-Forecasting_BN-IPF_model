# Data Presentation

The code is with a publicly and full-scale dataset, that can be download here: [Data]{https://doi.org/10.17632/dn6ng69xvr.1}. The code is designed to generate a population at the **individual-level**. 



## Training data size

The size of the available data varies depending on the data providers. 
In the published data, two data scenarios are presented that differ by the size of the training dataset:

- At 0.03% of the total population, this scenario mimics cases where the amount of data is minimal (Household Travel Survey at the national level) 0
- At 1% of the total population, this scenario corresponds to a Household Travel Survey at a narrower geographical scale, or Census data thourgh Public Use Microdata Sample (PUMS).


## Attributes Scenarios


Different data scenarios are provided to test different model complexities. 

| Category               | Variable          | Type     | 5 variables | 7 variables | 9 variables | 11 variables |
|------------------------|-------------------|----------|-------------|-------------|-------------|--------------|
| Person attributes      | Age               | int      | ✓           | ✓           | ✓           | ✓            |
|                        | Sex               | binary   | ✓           | ✓           | ✓           | ✓            |
|                        | Diploma           | category | ✓           | ✓           | ✓           | ✓            |
|                        | isMarried         | category |             |             | ✓           | ✓            |
|                        | Socioprofessional | category |             | ✓           | ✓           | ✓            |
|                        | Activity          | category |             |             |             | ✓            |
| Household attributes   | HouseholdSize     | int      | ✓           | ✓           | ✓           | ✓            |
|                        | nCars             | int      |             | ✓           | ✓           | ✓            |
|                        | Accommodation     | category |             |             | ✓           | ✓            |
|                        | Household         | category |             |             |             | ✓            |
| Geographical attribute | Department        | category | ✓           | ✓           | ✓           | ✓            |

# Note
- A data paper is in preparation for further details on the data collection, processing, and how to use it.