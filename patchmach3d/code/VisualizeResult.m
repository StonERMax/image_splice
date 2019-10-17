function VisualizeResult(V,map)

DET=squeeze(sum(sum(map)));
if sum(DET)>0
    DET=DET/max(DET);
end
h=figure;
for index=1:1:size(V,3)
    subplot(2,1,1)
    bar(1:size(V,3),DET,'r','EdgeColor','r'); hold on
    bar(1:size(V,3),1:size(V,3)==index,'b','EdgeColor','none'); hold off
    title(sprintf('Frame %d / %d',index,size(V,3)))
    axis([1 size(V,3) 0 1])
    subplot(2,1,2)
    FR(:,:,1)=V(:,:,index);
    FR(:,:,2)=V(:,:,index).*(1-map(:,:,index));
    FR(:,:,3)=V(:,:,index).*(1-map(:,:,index));
    imagesc(FR); axis image;
    if sum(sum(map(:,:,index)))==0
        pause(1/30)
    else
        pause(1/5)
    end
end

