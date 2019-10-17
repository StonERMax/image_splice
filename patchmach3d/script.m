clear all; close all; clc;
addpath code

%path = 'forge_frame';
path='vid';

V = read_frames(path);
size(V)

%% 
fprintf('Copy-Move Forgery Detection and Localization: Feature3D, fast approach\n');
[map, Vdet, timeFE, timePM, timePP]=CopyMoveForgeryDetection_Feature3D_fast(V);

V_map = V * cast(map, 'like', V);

for i = size(V, 2);

fprintf('\n');
