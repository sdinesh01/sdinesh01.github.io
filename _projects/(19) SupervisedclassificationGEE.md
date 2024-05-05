name: Random forest classification in GEE
tools: [Google Earth Engine]
image: https://github.com/sdinesh01/sdinesh01.github.io/blob/master/_images/myclassification_1.png?raw=true
description: Where I try my hand at a simple supervised classification of the Research Triangle in North Carolina on GEE (Sentinel-2)!
carousels:
  - images: 
    - image: https://github.com/sdinesh01/sdinesh01.github.io/blob/master/_images/myclassification_1.png?raw=true
    - image: https://github.com/sdinesh01/sdinesh01.github.io/blob/master/_images/nlcd_1.png?raw=true
---

# Random Forest Classification

This exercise was for GIS410 Intro to Remote Sensing at William & Mary. Ultimately, my simple five-class classification does not hold a candle to the National Land Cover Dataset but was a fun intro to Google Earth Engine. 

### Create training data
Place points on different land cover types for the forest, grass, builtup, Water, and barren land areas. We did this on top of an NDVI layer to easily detect different types of vegetation cover.

![search](https://github.com/sdinesh01/sdinesh01.github.io/blob/master/_images/classified_map_withpoints.png?raw=true)

### Evaluate the classification (accuracy score)

Here's a quick screenshot of the GEE console with model accuracy scores. 
![search](https://github.com/sdinesh01/sdinesh01.github.io/blob/master/_images/train_test_acc.png?raw=true)

### My quick map vs. the NCLD

{% include elements/carousel.html height="70" unit="%" number="1" %}



