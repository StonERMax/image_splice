function V=read_frames(path)
  files = dir(path);
  V = [];
  for i = 1:length(files)
      fp = char(fullfile(path, files(i).name));
      [~, ~, ext] = fileparts(fp);
      if strcmp(ext, '.jpg') | strcmp(ext, '.png') 
        try
          im = double(rgb2gray(imread(fp)))/255;
        catch err
          disp(['error reading ' fp]);
          continue;
        end
        V = cat(3, V, im);
      end
  end

%V=V/max(V(:));

end




