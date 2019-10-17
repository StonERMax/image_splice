clear all; close all; clc;
addpath code

pathVideo='EXEMPLEvideoFORGED.avi';
% pathVideo='EXEMPLEvideoPRISTINE.avi';
%pathVideo='./vid_forge.avi'


%% Read video
fprintf('Reading Video...\n');
V = readVideo(pathVideo);


%% 
fprintf('Copy-Move Forgery Detection and Localization: Feature3D, fast approach\n');
[map, Vdet, timeFE, timePM, timePP]=CopyMoveForgeryDetection_Feature3D_fast(V);
fprintf('\n');
%{
fprintf('Copy-Move Forgery Detection and Localization: Feature3D, basic approach\n');
[map, Vdet, timeFE, timePM, timePP]=CopyMoveForgeryDetection_Feature3D_basic(V);
fprintf('\n');

fprintf('Copy-Move Forgery Detection and Localization: Feature2D, fast approach\n');
[map, Vdet, timeFE, timePM, timePP]=CopyMoveForgeryDetection_Feature2D_fast(V);
fprintf('\n');

fprintf('Copy-Move Forgery Detection and Localization: Feature3D, basic approach\n');
[map, Vdet, timeFE, timePM, timePP]=CopyMoveForgeryDetection_Feature2D_basic(V);
fprintf('\n');
%}
%% 
%VisualizeResult(V,map);
