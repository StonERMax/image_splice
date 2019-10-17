clear all; close all; clc;
addpath code

dataset_path = '/home/islama6a/dataset/video_forge/davis_tempered/vid';

write_dir = 'davis';

if ~exist(write_dir)
    mkdir(write_dir);
end

write_dir_map = 'map_davis';

if ~exist(write_dir_map)
    mkdir(write_dir_map);
end

file_test_list = '../split/davis_test.txt';
file_list = read_file_into_array(file_test_list);

for i = 1:length(file_list)
    path = char(fullfile(dataset_path, file_list{i}));
    V = read_frames(path);
    fprintf('%d : %s\n', i, path);
    [map, Vdet, timeFE, timePM, timePP] = CopyMoveForgeryDetection_Feature3D_fast(V);
    fprintf('\n');

    wpath = char(fullfile(write_dir, file_list{i}));
    fprintf('   sum: %d\n', sum(map(:)));

    if ~exist(wpath)
        mkdir(wpath);
    end

    map = double(map > 0);
    nV = V .* map;

    [~, ~, c] = size(V);

    for j = 1:c
        im = uint8(nV(:, :, j) * 255);
        imwrite(im, fullfile(wpath, sprintf('%d.png', j)))
    end

    wpath2 = char(fullfile(write_dir_map, file_list{i}));
    save([wpath2 '.mat'], 'map');

end