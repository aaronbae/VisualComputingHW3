% YOU CAN RUN THE CODE BY EACH SECTION LIKE JUPYTER NOTEBOOK
%% Part 0 - Setup
clear all; close all; clc;
% add workpaths
addpath('code');
addpath('data');
run('vlfeat/toolbox/vl_setup'); % setup vlfeat library
% get image_file names
files = dir(fullfile('data','*.jpg'));
for i=1:length(files)
    name = convertCharsToStrings(files(i).name);
    image_files(i) = name;
end
% Get hand-crafted correspondence points
correspondence_matrix = get_correspondence_matrix();
disp("Part 0 finished");

%% Part 1: Extracting and matching features between images (20 points)
clc;
sift_results = get_sifts([1,2, 4], image_files);
disp("Part 1 finished");

%% Part 2: Finding the Camera Calibration Matrix (30 points)
clc;
c = calc_calibration(2, correspondence_matrix);
disp("Part 2 finished");

%% Part 3: Refining the Matching Featuers (20 points)
clc;
[F, e1, e2] = get_fundamental_matrix(1, 2, correspondence_matrix);
disp("Part 3 finished");

%% Part 4: Finding Depth (30 points)
clc;
disp("Part 4 finished");

%% Function Definitions
function sift_results=get_sifts(array_of_indices, image_files)
    sift_results=[];
    index_check = sum((array_of_indices < 0) + (array_of_indices > 12));
    if index_check ~= 0
        ME = MException('HW3:invalidInputIndex', 'All image indices must be between 0 and 12');
        throw(ME)
    end
    % get all the images
    for i=array_of_indices
        modified_index = i*2-1;
        image = getImage(modified_index, image_files);
        [F, D] = vl_sift(image);
        sift_results = [sift_results struct('index', i, 'image', image, 'F', F, 'D', D)];
        disp("Finished sift for image "+num2str(i));
    end
    % calculate sift for every adjacent pairs
    for curr_i=1:length(sift_results)
        next_i = curr_i + 1;
        if curr_i == length(sift_results)
            next_i = 1;
        end
        obj1 = sift_results(curr_i);
        obj2 = sift_results(next_i);
        [m, s] = vl_ubcmatch(obj1.D, obj2.D);
        obj1.m = m;
        obj2.s = s;
        
        % FURTHER FILTERING IS NOT IMPLEMENTED
        %[m, s] = sort_and_cutoff(unfiltered_m, unfiltered_s, 1.0);
        %[m, s] = filter_by_mask(mask1, mask2, F1, F2, m, s); 
        disp("Finished matching for image "+num2str(obj1.index)+" and "+num2str(obj2.index));
    end        
end
function [F, e1, e2]=get_fundamental_matrix(image_index_1, image_index_2, correspondence_matrix)
    % Summary:
    %   - returns the specified pairs of image's fundamental matrix
    % Parameters:
    %   - image_index_1: integer FROM 1 to 12
    %   - image_index_2: integer FROM 1 to 12
    %   - correspondence_matrix: the matrix from
    %   "get_correspondence_matrix" method
    % Returns: 
    %   - F: specified in "fundmatrix" function
    %   - e1: specified in "fundmatrix" function
    %   - e2: specified in "fundmatrix" function
    [points_3d, img1_points_2d]=get_correspondence_points(image_index_1, correspondence_matrix, false);
    [points_3d, img2_points_2d]=get_correspondence_points(image_index_2, correspondence_matrix, false);
    non_nan_index = sum(~isnan(img1_points_2d) + ~isnan(img2_points_2d),2) == 4;
    poitns_3d = points_3d(non_nan_index);
    img1_points_2d = img1_points_2d(non_nan_index,:);
    img1_points_2d = horzcat(img1_points_2d, ones(size(img1_points_2d, 1), 1)).';
    img2_points_2d = img2_points_2d(non_nan_index,:);
    img2_points_2d = horzcat(img2_points_2d, ones(size(img2_points_2d, 1), 1)).';
    [F, e1, e2] = fundmatrix(img1_points_2d, img2_points_2d);
end
function matrix=get_correspondence_matrix()
    % Summary:
    %   - returns the correspondence matrix that we pre-calcualted by hand
    %   - matrix has 20 rows. These are the 20 possible points in the image
    %   world.
    %   - matrix has 24 columns. These are the 12 different images with
    %   alternating X and Y values for each image.
    % Returns:
    %   - matrix: 20 x 24 matrix
    matrix = csvread('correspondence_points.csv');
    matrix = matrix(:, 1:end-1);
    matrix(matrix==0) = NaN;
end
function [points_3d, points_2d]=get_correspondence_points(image_index, correspondence_matrix, process_nan)
    % Summary:
    %   - returns the correspondence points for the specified image_index 
    % Parameters:
    %   - image_index: the index of the image. FROM 1 to 12
    %   - correspondence_matrix: the matrix from
    %   "get_correspondence_matrix" method
    % Returns:
    %   - points_3d: n x 3 matrix. n is the number of different points in
    %   3D
    %   - points_2d: n x 2 matrix. n is the number of different points in
    %   2D. Column 1 is X and Column 2 is Y
    points_3d = [
        0 0 19; % Blue Top
        64 0 19;
        64 64 19;
        0 64 19;
        0 0 29; % Red Top
        64 0 29;
        64 64 29;
        0 64 29;
        16 16 48; % Center Green Top
        48 16 48;
        48 48 48;
        16 48 48;
        0 48 48; % Corner Greeen Top
        32 48 48;
        32 80 48;
        0 80 48;
        0 48 67; % Yellow Top
        32 48 67;
        32 80 67;
        0 80 67];
    X = correspondence_matrix(:,image_index*2-1);
    Y = correspondence_matrix(:,image_index*2);
    if process_nan
        non_nan_index = ~isnan(X); % if X is Non-NaN, Y is Non-Nan
        X = X(non_nan_index);
        Y = Y(non_nan_index);
        points_3d = points_3d(non_nan_index, :);
    end
    points_2d = horzcat(X,Y);
    
    if process_nan
        % use only 8 points
        threshold = min(size(points_3d,1), 8);
        points_3d = points_3d(1:threshold, :);
        points_2d = points_2d(1:threshold, :);
    end
end
function c=calc_calibration(image_index, correspondence_matrix)
    % Summary:
    %   - returns the specified image's calibration matrix
    % Parameters:
    %   - image_index: integer FROM 1 to 12
    %   - correspondence_matrix: the matrix from
    %   "get_correspondence_matrix" method
    % Returns: 
    %   - C: 3 x 4 calibration matrix
    [points_3d, points_2d] = get_correspondence_points(image_index, correspondence_matrix, true);
    A = [];
    B = [];
    for i=1:length(points_3d)
        p_3d = points_3d(i,:);
        x = p_3d(1);
        y = p_3d(2);
        z = p_3d(3);
        w = 1;
        p_2d = points_2d(i,:);
        u = p_2d(1);
        v = p_2d(2);
        row1 = [-x -y -z -w  0  0  0  0 u*x u*y u*z];
        row2 = [ 0  0  0  0 -z -y -z -w v*x v*y v*z];
        A = [A; row1; row2];
        B = [B; -u; -v];
    end
    warning('off','all');
    c = linsolve(A, B);
    warning('on','all');
    c = [c; 1];
    c = reshape(c, [4,3]).';
end
function sort_descend_or_ascend_test(img1, img2, F1, F2, unfiltered_m, unfiltered_s)
    % Summary:
    %   - visualizes 9 different sets of features, categorized by scores
    %   - goal is to determine which sorting, ascend or descend, is better
    %   for sort_and_cutoff method
    % Parameters:
    %   - img1: image 1
    %   - img2: image 2
    %   - F1: Feature matrix for image 1
    %   - F2: Feature matrix for image 2
    %   - unfiltered_m: 2 x n matrix with each column representing the
    %   index of F1 and F2
    %   - unfiltered_s: 1 x n matrix with each representing a cooresponding
    %   score for each match in "unfiltered_m"
    [m, s] = sort_and_cutoff(unfiltered_m, unfiltered_s, 1.0);
    for i=1:9
        filter = (i-1)*100+1:(i*100)+1;
        temp_m = m(:,filter);
        visualize(img1, img2, F1, F2, temp_m);
    end
end
function [m,s]=filter_by_mask(mask1, mask2, F1, F2, oldM, oldS)
    % Summary:
    %   - filters features using the mask
    %   - currently showing just the black parts (pixel==0) of the mask
    % Parameters:
    %   - mask1: mask for image 1. Just another image
    %   - mask2: mask for image 2. Just another image
    %   - F1: Feature matrix for image 1
    %   - F2: Feature matrix for image 2
    %   - oldM: 2 x n matrix with each column representing the
    %   index of F1 and F2
    %   - oldS: 1 x n matrix with each representing a cooresponding
    %   score for each match in "unfiltered_m"
    m = [];
    s= [];
    for i=1:length(oldM)
        match = oldM(:,i);
        p1 = floor(F1(1:2, match(1)));
        p2 = floor(F2(1:2, match(2)));
        if ~mask1(p1(2), p1(1)) && ~mask2(p2(2),p2(1))
            m = [m match];
            s = [s oldS(i)];
        end
    end
end
function visualize(img1, img2, F1, F2, m)
    % Summary:
    %   - plots the image and the scatter of the feature centers
    % Parameters:
    %   - img1: image 1
    %   - img2: image 2
    %   - F1: Feature matrix for image 1
    %   - F2: Feature matrix for image 2
    %   - m: 2 x n matrix with each column representing the
    %   index of F1 and F2
    sz = 30;

    figure();
    subplot(1, 2, 1);
    imshow(img1)
    hold on
    scatter(F1(1,m(1,:)), F1(2,m(1,:)), sz, 'r', 'filled')

    subplot(1, 2, 2);
    imshow(img2)
    hold on
    scatter(F2(1,m(2,:)), F2(2,m(2,:)), sz, 'm', 'filled')
    ha=get(gcf,'children');
    set(gcf, 'position', [80 180 1424 534])
    set(ha(1),'position',[0 0 .5 1])
    set(ha(2),'position',[.5 0 .5 1])
end
function [newM, newS]=sort_and_cutoff(m, s, percentage)
    % Summary:
    %   - sorts the features by score and removes the poorest percentage
    % Parameters:
    %   - m: 2 x n matrix with each column representing the
    %   index of F1 and F2
    %   - s: 1 x n matrix with each element representing the score
    %   - percentage: fraction of poorest features to remove
    index = round(percentage*length(m));
    
    [newS, s_order] = sort(s,'ascend'); % which is better 'ascend' vs 'descend'?
    newM = m(:,s_order);
    newM = newM(:,1:index);
    newS = newS(1:index);
end
function img = getImage(index, image_files)
    % Summary:
    %   - retrieves the given image by index
    %   - odds are images and evens are masks
    % Parameters:
    %   - index: a number 
    %   - image_files: the list of file names in the '/data/' folder
    img = im2single( rgb2gray(imread( convertStringsToChars(image_files(index)) ) ) );
end
