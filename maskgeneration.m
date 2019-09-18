function [L, Ln] = maskgeneration(I, anns)

    [r, c, ~] = size(I);
    out = zeros(r, c);
    L = out;
    Ln = 0;

    n = length(anns); if(n == 0), return; end
    
    if (any(isfield(anns, {'segmentation', 'bbox'})))

        if (~isfield(anns, 'iscrowd')), [anns(:).iscrowd] = deal(0);
        end

        if (~isfield(anns, 'segmentation')), S = {anns.bbox}; %#ok<ALIGN>

            for i = 1:n, x = S{i}(1); w = S{i}(3); y = S{i}(2); h = S{i}(4);
                anns(i).segmentation = {[x, y, x, y + h, x + w, y + h, x + w, y]};
            end
        end

        S = {anns.segmentation}; hs = zeros(10000, 1); k = 0; hold on;
        pFill = {'FaceAlpha', .4, 'LineWidth', 3};

        for i = 1:n
            if (anns(i).iscrowd), C = [.01 .65 .40]; else C = rand(1, 3); end

            if (isstruct(S{i})) 
                M = double(MaskApi.decode(S{i}));
                out = out + M;
            else 
                P = MaskApi.frPoly(S{i}, r, c);
                M = double(MaskApi.decode(P));
                out = out + M;
            end

        end

    end

    out = double(out > 0);
    [L, Ln] = bwlabel(out);

end
