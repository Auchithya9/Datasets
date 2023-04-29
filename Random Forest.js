// A Sentinel-2 surface reflectance image, reflectance bands selected,
// serves as the source for training and prediction in this contrived example.
var img = ee.Image('ee.ImageCollection("COPERNICUS/S2_SR');

// ESA WorldCover land cover map, used as label source in classifier training.
var lc = ee.Image('ESA/WorldCover/v100/2020');

// Remap the land cover class values to a 0-based sequential series.
var classValues = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];
var remapValues = ee.List.sequence(0, 10);
var label = 'lc';
lc = lc.remap(classValues, remapValues).rename(label).toByte();


var roi = ee.Geometry.Polygon(
        [[[40.510539368536755, 37.39139948607018],
          [40.472087220099255, 37.281118661479105],
          [40.591262607780855, 37.15802302958226],
          [40.83708169957773, 37.179909368093256],
          [40.909077039111786, 37.33600680260806]]]);
// var roi = ee.Geometry.Rectangle(-122.347, 37.743, -122.024, 37.838);
var sample = img.addBands(lc).stratifiedSample({
  numPoints: 100,
  classBand: label,
  region: roi,
  scale: 10,
  geometries: true
});


sample = sample.randomColumn();
var trainingSample = sample.filter('random <= 0.8');
var validationSample = sample.filter('random > 0.8');

// Train a 10-tree random forest classifier from the training sample.
var trainedClassifier = ee.Classifier.smileRandomForest(10).train({
  features: trainingSample,
  classProperty: label,
  inputProperties: img.bandNames()
});


// Classify the reflectance image from the trained classifier.
var imgClassified = img.classify(trainedClassifier);

// Add the layers to the map.
var classVis = {
  min: 0,
  max: 10,
  palette: ['006400' ,'ffbb22', 'ffff4c', 'f096ff', 'fa0000', 'b4b4b4',
            'f0f0f0', '0064c8', '0096a0', '00cf75', 'fae6a0']
};
// Map.setCenter(-122.184, 37.796, 12);
Map.addLayer(img.clip(roi), {bands: ['B11', 'B8', 'B3'], min: 100, max: 3500}, 'img');
Map.addLayer(lc.clip(roi), classVis, 'lc');
Map.addLayer(imgClassified, classVis, 'Classified');
Map.addLayer(roi, {color: 'white'}, 'ROI', false, 0.5);
Map.addLayer(trainingSample, {color: 'black'}, 'Training sample', false);
Map.addLayer(validationSample, {color: 'white'}, 'Validation sample', false);

Export.image.toDrive({
   image: lc.clip(roi),
   description: 'Turkey_RF',
   folder:'Turkey',
   scale: 30,
   region: roi
  
   });
