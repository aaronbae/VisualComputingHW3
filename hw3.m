% YOU CAN RUN THE CODE BY EACH SECTION LIKE JUPYTER NOTEBOOK
%% Part 0 - Setup
clear all; close all; clc;
addpath('code');
addpath('data');
run('vlfeat/toolbox/vl_setup')
files = dir(fullfile('data','*.jpg'));
for i=1:length(files)
    name = convertCharsToStrings(files(i).name);
    image_files(i) = name;
end
disp("Part 0 finished");


%% Part 1: Extracting and matching features between images (20 points)
clc;
% get the first image and extract features
img1 = getImage(43, image_files);
mask1 = getImage(44, image_files);
[F1, D1] = vl_sift(img1);
size(F1)

% get the second image and extract features
img2 = getImage(45, image_files);
mask2 = getImage(46, image_files);
[F2, D2] = vl_sift(img2);
size(F2)

% match the features : THIS PART DOESN'T WORK WELL
[unfiltered_m, unfiltered_s] = vl_ubcmatch(D1, D2, 2.0);
% sort_and_cutoff isn't improving the results much, so leave it commented out for now
%[m, s] = sort_and_cutoff(unfiltered_m, unfiltered_s, 1.0);
%[m, s] = filter_by_mask(mask1, mask2, F1, F2, m, s); 
[m, s] = filter_by_mask(mask1, mask2, F1, F2, unfiltered_m, unfiltered_s); 

visualize(img1, img2, F1, F2, m);
disp("Part 0 finished");

%% Testing
clc;
[m, s] = sort_and_cutoff(unfiltered_m, unfiltered_s, 0.05);
[m, s] = filter_by_mask(mask1, mask2, F1, F2, m, s);
visualize(img1, img2, F1, F2, m);
%sort_descend_or_ascend_test(img1, img2, F1, F2, unfiltered_m, unfiltered_s);
disp("Testing finished");


%% Part 2: Finding the Camera Calibration Matrix (30 points) NOT FINISHED
clc;
% Hand Engineered matrix values
points_3d = [0 0 19; 64 0 19; 0 0 29; 64 0 29; 64 64 29; 16 16 48; 48, 16, 48; 48, 48, 48];
points_2d = [654 1902; 1569 1835; 637 1802; 1579 1731; 1453 1183; 872 1409; 1325, 1382; 1285, 1120];
c = calc_calibration(points_3d, points_2d);

for i=1:length(points_3d)
    p1 = [points_3d(i,:) 1].';
    result = (c*p1);
    result = round((result / result(3)).');
    p1 = p1.';
    fprintf("%2d ", p1);
    fprintf("  =>  ", p1);
    fprintf("%6d ", result);
    fprintf("  vs  ", p1);
    fprintf("%6d ", points_2d(i,:));
    disp(" ");
end


%% Function Definitions
function c=calc_calibration(points_3d, points_2d)
    % Summary:
    %   - CURRENTLY RETURNING TRIVIAL MATRIX (NEEDS A FIX)
    %   - returns a calibration matrix C from selected calibration points
    %   - look at page 162 of the textbook for how to derive the linear
    %   system
    % Parameters:
    %   - points_3d: n x 3 matrix with each row representing a 3D coordinate
    %   - points_2d: n x 2 matrix with each row representing a 2D coordinate
    % (pixel)
    % Returns: 
    %   - C: 3 x 4 calibration matrix
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
    c = linsolve(A, B);
    c = [c; 1];
    c = reshape(c, [3,4]);
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
