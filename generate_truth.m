% Create a data source from an image
imageDir = fullfile('training-images');
dataSource = groundTruthDataSource(imageDir);

% Define labels used to specify ground truth
names = {'Red_Mark'};
types = labelType('Rectangle');
labelDefs = table(names, types, 'VariableNames', {'Name', 'Type'});

% Determine number of images
numImages = numel(dataSource.Source);
Truth = cell(numImages, 1);

% Open JSON file containing coordinates
jsonData = jsondecode(fileread('coordinates.json'));

% Identify ROI in all images
for image = 1:numImages
    % Determine image name
    [filepath, name, ext] = fileparts(dataSource.Source{image});
    image_name = strrep([name ext],'.', '_');
    
    % Get data from JSON file
    coordinates = jsonData.(image_name).coordinates;
    coordinates_matrix = str2num(coordinates);
    
    % Add truth data
    Truth{image} = coordinates_matrix;
end

% Construct a table of label data
labelData = table(Truth, 'VariableNames', names);

% Create a groundTruth object
gTruth = groundTruth(dataSource, labelDefs, labelData);
disp(gTruth);

% Save groundTruth object
save gTruthData.mat gTruth