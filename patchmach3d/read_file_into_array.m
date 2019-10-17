function tlines = read_file_into_array(filename)
    fid = fopen(filename);
    tline = fgetl(fid);
    tlines = cell(0, 1);

    while ischar(tline)
        tlines{end + 1, 1} = tline;
        tline = fgetl(fid);
    end

    fclose(fid);
end
    
