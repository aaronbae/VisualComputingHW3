classdef ImageFile
    %IMAGEFILE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Name
        Image
        SiftFrame
        Descriptor
    end
    
    methods
        function obj = ImageFile(file_name)
            obj.Name = file_name;
            obj.Image = im2single( rgb2gray(imread( convertStringsToChars(file_name) ) ) );
            [ftemp, dtemp] = vl_sift(obj.Image);
            obj.SiftFrame = ftemp;
            obj.Descriptor = dtemp;
        end
    end
end

