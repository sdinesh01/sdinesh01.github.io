---
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

### code for this analysis

```javascript
/*set up function to mask clouds using the Sentinel-2 QA band*/
function maskS2clouds(image) {
 
  // select QA band for Sentinel-2
  var qa = image.select('QA60');
 
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
 
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
  	.and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  	
  // Mask the cloudy image and scale the unit by dividing 10000
  return image.updateMask(mask).divide(10000);
}
 
//lat & lon for RTP
var point = ee.Geometry.Point(-79.04, 35.91);
 
// Load Sentinel-2 surface reflectance data and
// filter the images with time, place and cloud conditions
var dataset = ee.ImageCollection('COPERNICUS/S2_SR')
              	.filterDate('2020-05-01', '2020-10-30')
              	.filterBounds(point) //filter by location
              	// Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
              	.map(maskS2clouds);
         	
var image = dataset.first();
var rgbVis = {
  min: 0.0,
  max: 0.2,
  bands: ['B4', 'B3', 'B2'],
};
 
Map.centerObject(image,10);
//load both the true color and the color infrared
Map.addLayer(image, rgbVis, 'Sentinel-2 RGB');
Map.addLayer(image, {bands:['B8', 'B4', 'B3'], min:0, max:1}, 'Sentinel-2 CIR');

// calculate NDVI and add it to image
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
image = image.addBands(ndvi)
//display the NDVI
Map.addLayer(ndvi, {min: 0.0, max: 1}, "NDVI");

// specify the feature list used in the following
var feature_bands = ['B2','B3','B4','B8','NDVI'];

// Classification training samples were obtained by hand-drawn geometry  
// Merge samples together
var feature_samples = Forest.merge(Grass).merge(Builtup).merge(Barren).merge(Water);

// add a column of random numbers [0-1]
feature_samples = feature_samples.randomColumn('random',1);
 
// Randomly split the data into 70% for training, and 30% for testing
 
var training_data =feature_samples.filter(ee.Filter.lte('random', 0.70));
var testing_data =feature_samples.filter(ee.Filter.gt('random', 0.70));

// Sample the pixels of the input imagery to get a FeatureCollection of training data.
var training = image.select(feature_bands).sampleRegions({
  collection: training_data, //
  properties: ['landcover'], // The list of properties to copy from each input feature
  scale: 10  // pixel scale
});
 
var testing = image.select(feature_bands).sampleRegions({
  collection: testing_data, //
  properties: ['landcover'], // The list of properties to copy from each input feature
  scale: 10  // pixel scale
});

// Make a Random Forest classifier and train it.
var trained_classifier = ee.Classifier.smileRandomForest(10) // default parameters for random forest and number of trees set to 10
  .train({
  features: training,
  classProperty: 'landcover',
  inputProperties: feature_bands
});

// Training accuracy evaluation
var trainAccuracy = trained_classifier.confusionMatrix();
print('Overall Accuracy for Training: ', trainAccuracy.accuracy());

// Testing accuracy evaluation
// Perform the Random Forest on the testing data
var RFtesting = testing
    .classify(trained_classifier, 'predict');
 
// calculate error matrix based on actual and predicted land cover types
// only type values in the list [1,2,3,4,5] are used
var testAccuracy = RFtesting.errorMatrix('landcover','predict',[1,2,3,4,5])
print('Overall Accuracy for Testing: ', testAccuracy.accuracy());

// Classify the input imagery.
var classified_image = image.select(feature_bands).classify(trained_classifier);

// Define a palette for the land classification.
var palette = [
   '008000', // Forest (1)
  '8df4ff', // Grass (2)
  'f56952', // Buildup (3)
  '9d7d73', // Barren (4)
  '514eff'// Water (5)
];
// plot your classification
Map.addLayer(classified_image, {min: 1, max: 5, palette: palette}, 'Classification');

// Import the NLCD collection.
var dataset = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD');
// The collection contains images for the 2021 year release and the full suite
// of products.
print('Products:', dataset.aggregate_array('system:index'));
// Filter the collection to the 2021 product.
var nlcd2021 = dataset.filter(ee.Filter.eq('system:index', '2021')).first();
// Each product has multiple bands for describing aspects of land cover.
print('Bands:', nlcd2021.bandNames());
// Select the land cover band.
var landcover = nlcd2021.select('landcover');
// Display land cover on the map.
Map.setCenter(-79.04, 35.91, 9);
Map.addLayer(landcover, null, 'NLCD Landcover 2021');

// Add classification legend
// set position of panel
var legend = ui.Panel({
  style: {
	position: 'bottom-left',
	padding: '8px 15px'
  }
});
 
// Create legend title
var legendTitle = ui.Label({
  value: 'Legend',
  style: {
	fontWeight: 'bold',
	fontSize: '18px',
	margin: '0 0 4px 0',
	padding: '0'
	}
});
 
// Add the title to the panel
legend.add(legendTitle);
	
// Creates and styles 1 row of the legend.
var makeRow = function(color, name) {
  	
  	// Create the label that is actually the colored box.
  	var colorBox = ui.Label({
    	style: {
          backgroundColor: '#' + color,
      	// Use padding to give the box height and width.
      	padding: '8px',
      	margin: '0 0 4px 0'
    	}
  	});
  	
  	// Create the label filled with the description text.
  	var description = ui.Label({
    	value: name,
    	style: {margin: '0 0 4px 6px'}
  	});
  	
  	// return the panel
  	return ui.Panel({
    	widgets: [colorBox, description],
    	layout: ui.Panel.Layout.Flow('horizontal')
  	});
};
 
// Add color and name
 
var classType = ['Forest','Grass','Built-up','Barren','Water']
 
classType.forEach(function(ID, index) {
legend.add(makeRow(palette[index], classType[index]));
});
 
// Add legend to map
Map.add(legend);

Export.image.toDrive(classified_image, 'classificationlab');
```



