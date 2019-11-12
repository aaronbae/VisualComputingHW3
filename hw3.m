%% Part 0 - Setup
clear all; close all; clc;
addpath('code');
addpath('data');
run('vlfeat/toolbox/vl_setup')
files = dir(fullfile('data','*.jpg'));
for i=1:length(files)
    name = convertCharsToStrings(files(i).name);
    %fileIndex = ceil(i / 2); 
    %maskIndex = mod(i, 2) + 1;
    %image_files(fileIndex, maskIndex) = name;
    image_files(i) = name;
end
disp("Part 0 finished");


%% Part 1: Extracting and matching features between images (20 points)
clc;
img1 = getImage(43, image_files);
mask1 = getImage(44, image_files);
[F1, D1] = vl_sift(img1);
size(F1)

img2 = getImage(45, image_files);
mask2 = getImage(46, image_files);
[F2, D2] = vl_sift(img2);
size(F2)

[unfiltered_m, unfiltered_s] = vl_ubcmatch(D1, D2, 2.0);
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


%% Part 2: Finding the Camera Calibration Matrix (30 points)
clc;
points_3d = [0 0 19; 64 0 19; 0 0 29; 64 0 29; 64 64 29; 16 16 48];
points_2d = [654 1902; 1569 1835; 637 1802; 1579 1541; 1452 1183; 872 1409];
c = calc_calibration(points_3d, points_2d);
disp(c);


%% Function Definitions
function c=calc_calibration(points_3d, points_2d)
    A = [];
    for i=1:length(points_3d)
        p_3d = points_3d(i,:);
        p_2d = points_2d(i,:);
        u = p_2d(1);
        v = p_2d(2);
        row1 = [-p_3d(1) -p_3d(2) -p_3d(3) -1 0 0 0 0 u*p_3d(1) u*p_3d(2) u*p_3d(3) u];
        row2 = [0 0 0 0 -p_3d(1) -p_3d(2) -p_3d(3) -1 v*p_3d(1) v*p_3d(2) v*p_3d(3) v];
        A = [A; row1; row2];
    end
    B = zeros(size(A, 1),1);
    c = linsolve(A, B);
end
function sort_descend_or_ascend_test(img1, img2, F1, F2, unfiltered_m, unfiltered_s)
    [m, s] = sort_and_cutoff(unfiltered_m, unfiltered_s, 1.0);
    for i=1:9
        filter = (i-1)*100+1:(i*100)+1;
        temp_m = m(:,filter);
        temp_s = s(filter);
        visualize(img1, img2, F1, F2, temp_m);
    end
end
function [m,s]=filter_by_mask(mask1, mask2, F1, F2, oldM, oldS)
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

function [newM, newS]=sort_and_cutoff(m, s, quant)
    index = round(quant*length(m));
    
    [newS, s_order] = sort(s,'ascend');
    newM = m(:,s_order);
    newM = newM(:,1:index);
    newS = newS(1:index);
end

function img = getImage(index, image_files)
    img = im2single( rgb2gray(imread( convertStringsToChars(image_files(index)) ) ) );
end
