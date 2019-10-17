clear all; close all; clc;
addpath code

dataset_path = '/home/islama6a/dataset/video_forge/tmp_youtube_tempered/vid';

%dataset_path = '/home/islama6a/dataset/video_forge/tmp/vid';
dat_dir = get_valid_dir(dataset_path);
%dat_dir = {'9_cow_0016'};

write_dir = 'tmp_utube';

if ~exist(write_dir)
    mkdir(write_dir);
end

write_dir2 = 'tmp2_utube';

if ~exist(write_dir2)
    mkdir(write_dir2);
end

for i = 1:length(dat_dir)
    path = char(fullfile(dataset_path, dat_dir{i}));
    V = read_frames(path);
    fprintf('%d : %s\n', i, path);
    [map, Vdet, timeFE, timePM, timePP] = CopyMoveForgeryDetection_Feature3D_fast(V);
    fprintf('\n');

    wpath = char(fullfile(write_dir, dat_dir(i)));
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

    wpath2 = char(fullfile(write_dir2, dat_dir(i)));
    save([wpath2 '.mat'], 'map');

end
