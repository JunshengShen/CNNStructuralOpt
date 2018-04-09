X=load('lable50.txt');

X=(reshape(X(1:16384),128,128));

imagesc(X),colorbar,colormap gray;