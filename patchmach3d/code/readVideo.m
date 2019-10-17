function V = readVideo(pathVideo)
OBJ=VideoReader(pathVideo);
V=zeros(OBJ.Height,OBJ.Width,OBJ.NumberOfFrames);
for k= 1 : OBJ.NumberOfFrames
    t = read(OBJ,k);
    V(:,:,k) = rgb2gray(t);
end
V=V/max(V(:));
OBJ.delete;