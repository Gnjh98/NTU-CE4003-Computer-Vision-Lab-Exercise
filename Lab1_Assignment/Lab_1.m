%Lab 1 Assignment

%% Q2.1a: input image into MATLAB matrix variable

Pc = imread('mrt-train.jpg');
whos Pc
P = rgb2gray(Pc);

%% Q2.1b: Viewing the image

figure;imshow(P);

%% Q2.1c: Check min and max intensities present in image

min(P(:)),max(P(:))

%% Q2.1d: code for contrast stretching

x = double(P);
y = x - 13;
z = (255/(204-13))*y;
P2 = uint8(z);

min(P2(:)),max(P2(:))

%% Q2.1e: Redisplay image P2
figure;imshow(P2);

%% Q2.2a: Display image intensity of histogram P

%using 10 bins
figure;imhist(P,10);

%using 256bins
figure;imhist(P,256);

%% Q2.2b: histogram of P3

P3 = histeq(P,255);
figure;imhist(P3,10);
figure;imhist(P3,255);

%% Q2.2c: rerun histogram equalization P3

P3 = histeq(P3,255);
figure;imhist(P3,10);
figure;imhist(P3,255);

%% Q2.3a Generate filter

norm = -2:1:2;
[X Y] = meshgrid(norm, norm)
h1 = (1/(2*pi*1*1))*exp(-((X.^2 + Y.^2)/(2*1.^2)));
h2 = (1/(2*pi*2*2))*exp(-((X.^2 + Y.^2)/(2*2.^2)));
h1 = h1/sum(h1(:))
h2 = h2/sum(h2(:))

%check sum of all elements=1
sumh1 = sum(h1(:))
sumh2 = sum(h2(:))

figure;mesh(double(h1));
figure;mesh(double(h2));

%% Q2.3b View ntu-gn.jpg

P4 = imread('ntu-gn.jpg');
figure;imshow(P4);

%% Q2.3c filtering the image

%h1 filter sigma=1
P4_double = double(P4);
P4_h1_filtered = conv2(double(h1), P4_double);
P4_h1 = uint8(P4_h1_filtered);
figure;imshow(P4_h1);

%h2 filter sigma=2
P4_h2_filtered = conv2(double(h2), P4_double);
P4_h2 = uint8(P4_h2_filtered);
figure;imshow(P4_h2);

%% Q2.3d View ntu-sp.jpg

P5 = imread('ntu-sp.jpg');
figure;imshow(P5);

%% Q2.3e Repeat step (c) for speckle noise

%h1 filter sigma=1
P5_double = double(P5);
P5_h1_filtered = conv2(double(h1), P5_double);
P5_h1 = uint8(P5_h1_filtered);
figure;imshow(P5_h1);

%h2 filter sigma=2
P5_h2_filtered = conv2(double(h2), P5_double);
P5_h2 = uint8(P5_h2_filtered);
figure;imshow(P5_h2);

%% Q2.4 Median filtering

%3x3 median filter on gaussian noise
P4_median = medfilt2(P4_double, [3 3]);
P4_median = uint8(P4_median);
figure;imshow(P4_median);

%5x5 median filter on gaussian noise
P4_median = medfilt2(P4_double, [5 5]);
P4_median = uint8(P4_median);
figure;imshow(P4_median);

% 3x3 median filter on speckle noise
P5_median = medfilt2(P5_double, [3 3]);
P5_median = uint8(P5_median);
figure;imshow(P5_median);

% 5x5 median filter on speckle noise
P5_median = medfilt2(P5_double, [5 5]);
P5_median = uint8(P5_median);
figure;imshow(P5_median);

%% Q2.5a View pck-int.jpg

P6 = imread('pck-int.jpg');
figure;imshow(P6);

%% Q2.5b Obtain fourier transform F and power spectrum S. Display the power spectrum

%Obtain Fourier transform and power spectrum
F = fft2(P6);
S = abs(F);

%Display power spectrum
imagesc(fftshift(S.^0.1));
colormap('default');

%% Q2.5c Redisplay power spectrum without fftshift

imagesc(S.^0.1);
colormap('default');

%locations of peaks
[x y]=ginput(2)
x1 = 9;
y1 = 241;
x2 = 249;
y2 = 17;

%% Q2.5d recompute power spectrum

%coordinates of peaks
x1 = 9;
y1 = 241;
x2 = 249;
y2 = 17;

%set 5x5 neighbourhood of peaks to zero
F(y1-2:y1+2, x1-2:x1+2) = zeros(5);
F(y2-2:y2+2, x2-2:x2+2) = zeros(5);

%recompute power spectrum and display
S = abs(F);
imagesc(S.^0.1);
colormap('default');

%% Q2.5e Compute inverse Fourier transform

P7 = uint8(ifft2(F));
figure;imshow(P7);

%% Q2.5f "free" primate-caged.jpg

%display image
P8 = imread("primate-caged.jpg");
P8 = rgb2gray(P8);
imshow(P8);

%display power spectrum
F2 = fft2(P8);
S = abs(F2);
imagesc(S.^0.1);
colormap('default');

%find location of peaks
%[x y] = ginput(4)
x1 = 11;
y1 = 252;
x2 = 4;
y2 = 21;
x3 = 247;
y3 = 4;
x4 = 237;
y4 = 10;

%set 5x5 neighbouring to 0
F2(y1-2:y1+2, x1-2:x1+2) = zeros(5);
F2(y2-2:y2+2, x2-2:x2+2) = zeros(5);
F2(y3-2:y3+2, x3-2:x3+2) = zeros(5);
F2(y4-2:y4+2, x4-2:x4+2) = zeros(5);

%recompute power spectrum and display
S = abs(F2);
imagesc(S.^0.1);
colormap('default');

%compute inverse Fourier transform and display
P8 = uint8(ifft2(F2));
figure;imshow(P8);

%% Q2.6a download and display book.jpg

P9 = imread('book.jpg');
figure;imshow(P9);

%% Q2.6b coordinates of 4 corners
figure;imshow(P9);
[x y]=ginput(4)


%% Q2.6c projective transformation

%clockwise direction starting from top left
x1 = 143;
y1 = 28;
x2 = 308;
y2 = 47;
x3 = 256;
y3 = 216;
x4 = 5;
y4 = 159;
x = [143;308;256;5];
y = [28;47;216;159];
%coordinates of output image
x_im1 = 0;
y_im1 = 0;
x_im2 = 210;
y_im2 = 0;
x_im3 = 210;
y_im3 = 297;
x_im4 = 0;
y_im4 = 297;


v = [0;0;210;0;210;297;0;297];
A = [x1,y1,1,0,0,0,-x_im1*x1,-x_im1*y1;
     0,0,0,x1,y1,1,-y_im1*x1,-y_im1*y1;
     x2,y2,1,0,0,0,-x_im2*x2,-x_im2*y2;
     0,0,0,x2,y2,1,-y_im2*x2,-y_im2*y2;
     x3,y3,1,0,0,0,-x_im3*x3,-x_im3*y3;
     0,0,0,x3,y3,1,-y_im3*x3,-y_im3*y3;
     x4,y4,1,0,0,0,-x_im4*x4,-x_im4*y4;
     0,0,0,x4,y4,1,-y_im4*x4,-y_im4*y4;];

u = A\v;
U = reshape([u;1],3,3)';

w = U*[x';y'; ones(1,4)];
w = w./(ones(3,1)*w(3,:))

%% Q2.6d Wrap image

T = maketform('projective', U');
P2 = imtransform(P9,T,'XData',[0 210],'YData',[0 297]);

%% Q2.6e display image
figure;imshow(P2);
%% Q2.6f identify pink area

%figure;imshow(P2);
%[x, y] = ginput(1)
x = 164;
y = 175;

R = P2(:,:,1);
G = P2(:,:,2);
B = P2(:,:,3);

P2_pink = P2;
[R(x,y) G(x,y) B(x,y)]

%Get the pink area, set the rest white
for cha = 1:size(P2_pink, 3)
    for row = 1:size(P2_pink,1)
        for col = 1:size(P2_pink,2)
            if P2_pink(row,col,1) >= 180 & P2_pink(row,col,1) <= 230 &  P2_pink(row,col,2) <= 160 & P2_pink(row,col,3) <= 160
            else
                P2_pink(row,col,1) = 255;
                P2_pink(row,col,2) = 255;
                P2_pink(row,col,3) = 255;
            end
        end
    end
end

figure;imshow(P2_pink);