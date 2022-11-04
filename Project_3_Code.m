clear all; close all; clc;
% Test 1: Ideal Case
load cam1_1.mat;  % -- Load Movie cam1_1 (Ideal Case from Cam 1)
[height1_1 width1_1 rgb num_frames1_1] = size(vidFrames1_1);
x_1_1 = [];
y_1_1 = [];
for j = 1:num_frames1_1  % -- Watch Movie
X = rgb2gray(vidFrames1_1(:,:,:,j));  % convert to grey scale 
%imshow(X); drawnow
X1 = double(X);  % converted to double precision for mathematical processing
X1(:,1:300) = 0;  % crop the video to the target region 
X1(:,400:end) = 0;
X1(1:200,:) = 0;
X1(450:end,:) = 0;
[M,I] = max(X1(:));
[y,x] = ind2sub(size(X1),I);
x_1_1 = [x_1_1 x];  % record the bright spot position in each frame
y_1_1 = [y_1_1 y];
end
load cam2_1.mat;  % -- Load Movie cam2_1 (Ideal Case from Cam 2)
[height2_1 width2_1 rgb num_frames2_1] = size(vidFrames2_1);
x_2_1 = [];
y_2_1 = [];
for j=1:num_frames2_1  % -- Watch Movie
X = rgb2gray(vidFrames2_1(:,:,:,j));  % convert to grey scale 
%imshow(X); drawnow
X2 = double(X);  % converted to double precision for mathematical processing
X2(:,1:250) = 0;  % crop the video to the target region 
X2(:,350:end) = 0;
X2(1:100,:) = 0;
X2(380:end,:) = 0;
[M,I] = max(X2(:));
[y,x] = ind2sub(size(X2),I);
x_2_1 = [x_2_1 x];  % record the bright spot position in each frame
y_2_1 = [y_2_1 y];
end
load cam3_1.mat;  % -- Load Movie cam3_1 (Ideal Case from Cam 3)
[height3_1 width3_1 rgb num_frames3_1] = size(vidFrames3_1);
x_3_1 = [];
y_3_1 = [];
for j=1:num_frames3_1  % -- Watch Movie
X = rgb2gray(vidFrames3_1(:,:,:,j));  % convert to grey scale 
%imshow(X); drawnow
X3 = double(X);  % converted to double precision for mathematical processing
X3(:,1:270) = 0;  % crop the video to the target region 
X3(:,480:end) = 0;
X3(1:230,:) = 0;
X3(340:end,:) = 0;
[M,I] = max(X3(:));
[y,x] = ind2sub(size(X3),I);
x_3_1 = [x_3_1 x];  % record the bright spot position in each frame
y_3_1 = [y_3_1 y];
end

% make all 3 cams start at same relative position
[Min,Ind] = min(y_1_1(1:25));
x_1_1 = x_1_1(Ind:end);
y_1_1 = y_1_1(Ind:end);
[Min,Ind] = min(y_2_1(1:25));
x_2_1 = x_2_1(Ind:end);
y_2_1 = y_2_1(Ind:end);
[Min,Ind] = min(x_3_1(1:25));
x_3_1 = x_3_1(Ind:end);
y_3_1 = y_3_1(Ind:end);
% trim each movie clip down to the same size
frames_min = min([length(x_1_1),length(x_2_1),length(x_3_1)]);
x_1_1 = x_1_1(1:frames_min);
x_1_1 = x_1_1 - mean(x_1_1);
y_1_1 = y_1_1(1:frames_min);
y_1_1 = y_1_1 - mean(y_1_1);
x_2_1 = x_2_1(1:frames_min);
x_2_1 = x_2_1 - mean(x_2_1);
y_2_1 = y_2_1(1:frames_min);
y_2_1 = y_2_1 - mean(y_2_1);
x_3_1 = x_3_1(1:frames_min);
x_3_1 = x_3_1 - mean(x_3_1);
y_3_1 = y_3_1(1:frames_min);
y_3_1 = y_3_1 - mean(y_3_1);
XY_total = [x_1_1; y_1_1; x_2_1; y_2_1; x_3_1; y_3_1];

figure(1)  % Plot the Raw Data Positional Data of Paint Can
subplot(3,1,1)
plot(1:frames_min, XY_total(2,:),1:frames_min, XY_total(1,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 1-Ideal Case: Raw Positional Data of Paint Can, Cam 1");
legend("Y", "X")
subplot(3,1,2)
plot(1:frames_min, XY_total(4,:),1:frames_min, XY_total(3,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 1-Ideal Case: Raw Positional Data of Paint Can, Cam 2");
legend("Y", "X")
subplot(3,1,3)
plot(1:frames_min, XY_total(6,:),1:frames_min, XY_total(5,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 1-Ideal Case: Raw Positional Data of Paint Can, Cam 3")
legend("Y", "X")

% PCA on test 1
[U,S,V] = svd(XY_total/sqrt(frames_min-1), 'econ');
sig = diag(S); 
figure(2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2)
title("Test 1-Ideal Case: Energy Captured by Each Nonzero Singular Value");
xlabel("Singular Values"); ylabel("Energy");
axis([0 6 0 1])
set(gca,'Fontsize',14,'Xtick',0:1:6)
% Create Projection on first three Principal Components
figure(3)
XY_proj = V';
plot(1:frames_min, XY_proj(1:2,:),'Linewidth',1.5)
title("Test 1-Ideal Case: Movements on Principal Components");
ylabel("Displacement"); xlabel("Time (frames)"); 
legend("1st Principal Component", "2nd Principal Component", ... 
       "3rd Principal Component",'location','southeast');
set(gca,'Fontsize',14)

%% Test 2: Noisy Case.
clear all; close all; clc;
load cam1_2.mat;  % -- Load Movie cam1_2 (Noisy Case from Cam 1)
[height1_2 width1_2 rgb num_frames1_2] = size(vidFrames1_2);
x_1_2 = [];
y_1_2 = [];
for j = 1:num_frames1_2  % -- Watch Movie
X = rgb2gray(vidFrames1_2(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X1 = double(X);  % converted to double precision for mathematical processing
X1(:,1:300) = 0;  % crop the video to the target region 
X1(:,410:end) = 0;
X1(1:220,:) = 0;
X1(450:end,:) = 0;
[M,I] = max(X1(:));
[y,x] = ind2sub(size(X1),I);
x_1_2 = [x_1_2 x];  % record the bright spot position in each frame
y_1_2 = [y_1_2 y];
end
load cam2_2.mat;  % -- Load Movie cam2_2 (Noisy Case from Cam 2)
[height2_2 width2_2 rgb num_frames2_2] = size(vidFrames2_2);
x_2_2 = [];
y_2_2 = [];
for j=1:num_frames2_2  % -- Watch Movie
X = rgb2gray(vidFrames2_2(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X2 = double(X);  % converted to double precision for mathematical processing
X2(:,1:200) = 0;  % crop the video to the target region 
X2(:,400:end) = 0;
X2(1:50,:) = 0;
X2(450:end,:) = 0;
[M,I] = max(X2(:));
[y,x] = ind2sub(size(X2),I);
x_2_2 = [x_2_2 x];  % record the bright spot position in each frame
y_2_2 = [y_2_2 y];
end
load cam3_2.mat;  % -- Load Movie cam3_2 (Noisy Case from Cam 3)
[height3_2 width3_2 rgb num_frames3_2] = size(vidFrames3_2);
x_3_2 = [];
y_3_2 = [];
for j=1:num_frames3_2  % -- Watch Movie
X = rgb2gray(vidFrames3_2(:,:,:,j));  % convert to grey scale 
%imshow(X); drawnow
X3 = double(X);  % converted to double precision for mathematical processing
X3(:,1:270) = 0;  % crop the video to the target region 
X3(:,480:end) = 0;
X3(1:200,:) = 0;
X3(340:end,:) = 0;
[M,I] = max(X3(:));
[y,x] = ind2sub(size(X3),I);
x_3_2 = [x_3_2 x];  % record the bright spot position in each frame
y_3_2 = [y_3_2 y];
end

% make all 3 cams start at same relative position
[Min,Ind] = min(y_1_2(1:25));
x_1_2 = x_1_2(Ind:end);
y_1_2 = y_1_2(Ind:end);
[Min,Ind] = min(y_2_2(1:25));
x_2_2 = x_2_2(Ind:end);
y_2_2 = y_2_2(Ind:end);
[Min,Ind] = min(x_3_2(1:25));
x_3_2 = x_3_2(Ind:end);
y_3_2 = y_3_2(Ind:end);
% trim each movie clip down to the same size
frames_min = min([length(x_1_2),length(x_2_2),length(x_3_2)]);
x_1_2 = x_1_2(1:frames_min);
x_1_2 = x_1_2 - mean(x_1_2);
y_1_2 = y_1_2(1:frames_min);
y_1_2 = y_1_2 - mean(y_1_2);
x_2_2 = x_2_2(1:frames_min);
x_2_2 = x_2_2 - mean(x_2_2);
y_2_2 = y_2_2(1:frames_min);
y_2_2 = y_2_2 - mean(y_2_2);
x_3_2 = x_3_2(1:frames_min);
x_3_2 = x_3_2 - mean(x_3_2);
y_3_2 = y_3_2(1:frames_min);
y_3_2 = y_3_2 - mean(y_3_2);
XY_total = [x_1_2; y_1_2; x_2_2; y_2_2; x_3_2; y_3_2];

figure(1)  % Plot the Raw Data Positional Data of Paint Can
subplot(3,1,1)
plot(1:frames_min, XY_total(2,:),1:frames_min, XY_total(1,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 2-Noisy Case: Raw Positional Data of Paint Can, Cam 1");
legend("Y", "X")
subplot(3,1,2)
plot(1:frames_min, XY_total(4,:),1:frames_min, XY_total(3,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 2-Noisy Case: Raw Positional Data of Paint Can, Cam 2");
legend("Y", "X")
subplot(3,1,3)
plot(1:frames_min, XY_total(6,:),1:frames_min, XY_total(5,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 2-Noisy Case: Raw Positional Data of Paint Can, Cam 3")
legend("Y", "X")

% PCA on test 2
[U,S,V] = svd(XY_total/sqrt(frames_min-1), 'econ');
sig = diag(S);  % Calculating the energy of the truncations
figure(2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2)
title("Test 2-Noisy Case: Energy Captured by Each Nonzero Singular Value");
xlabel("Singular Values"); ylabel("Energy");
axis([0 6 0 1])
set(gca,'Fontsize',14,'Xtick',0:1:6)
% Create Projection on first three Principal Components
figure(3)
XY_proj = V';
plot(1:frames_min, XY_proj(1:3,:),'Linewidth',1.5)
title("Test 2-Noisy Case: Movements on Principal Components");
ylabel("Displacement"); xlabel("Time (frames)"); 
legend("1st Principal Component", "2nd Principal Component", ... 
       "3rd Principal Component",'location','southeast');
set(gca,'Fontsize',14)
   
%% Test 3: Horizontal Displacement Case
clear all; close all; clc;
load cam1_3.mat;  % -- Load Movie cam1_3 (Horizontal Displacement Case from Cam 1)
[height1_3 width1_3 rgb num_frames1_3] = size(vidFrames1_3);
x_1_3 = [];
y_1_3 = [];
for j = 1:num_frames1_3  % -- Watch Movie
X = rgb2gray(vidFrames1_3(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X1 = double(X);  % converted to double precision for mathematical processing
X1(:,1:280) = 0;  % crop the video to the target region 
X1(:,400:end) = 0;
X1(1:225,:) = 0;
X1(380:end,:) = 0;
[M,I] = max(X1(:));
[y,x] = ind2sub(size(X1),I);
x_1_3 = [x_1_3 x];  % record the bright spot position in each frame
y_1_3 = [y_1_3 y];
end
load cam2_3.mat;  % -- Load Movie cam2_3 (Horizontal Displacement Case from Cam 2)
[height2_3 width2_3 rgb num_frames2_3] = size(vidFrames2_3);
x_2_3 = [];
y_2_3 = [];
for j=1:num_frames2_3  % -- Watch Movie
X = rgb2gray(vidFrames2_3(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X2 = double(X);  % converted to double precision for mathematical processing
X2(:,1:220) = 0;  % crop the video to the target region 
X2(:,400:end) = 0;
X2(1:180,:) = 0;
X2(380:end,:) = 0;
[M,I] = max(X2(:));
[y,x] = ind2sub(size(X2),I);
x_2_3 = [x_2_3 x];  % record the bright spot position in each frame
y_2_3 = [y_2_3 y];
end
load cam3_3.mat;  % -- Load Movie cam3_3 (Horizontal Displacement Case from Cam 3)
[height3_3 width3_3 rgb num_frames3_3] = size(vidFrames3_3);
x_3_3 = [];
y_3_3 = [];
for j=1:num_frames3_3  % -- Watch Movie
X = rgb2gray(vidFrames3_3(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X3 = double(X);  % converted to double precision for mathematical processing
X3(:,1:270) = 0;  % crop the video to the target region 
X3(:,480:end) = 0;
X3(1:170,:) = 0;
X3(320:end,:) = 0;
[M,I] = max(X3(:));
[y,x] = ind2sub(size(X3),I);
x_3_3 = [x_3_3 x];  % record the bright spot position in each frame
y_3_3 = [y_3_3 y];
end

% make all 3 cams start at same relative position
[Min,Ind] = min(y_1_3(1:25));
x_1_3 = x_1_3(Ind:end);
y_1_3 = y_1_3(Ind:end);
[Min,Ind] = min(y_2_3(1:25));
x_2_3 = x_2_3(Ind:end);
y_2_3 = y_2_3(Ind:end);
[Min,Ind] = min(x_3_3(1:25));
x_3_3 = x_3_3(Ind:end);
y_3_3 = y_3_3(Ind:end);
% trim each movie clip down to the same size
frames_min = min([length(x_1_3),length(x_2_3),length(x_3_3)]);
x_1_3 = x_1_3(1:frames_min);
x_1_3 = x_1_3 - mean(x_1_3);
y_1_3 = y_1_3(1:frames_min);
y_1_3 = y_1_3 - mean(y_1_3);
x_2_3 = x_2_3(1:frames_min);
x_2_3 = x_2_3 - mean(x_2_3);
y_2_3 = y_2_3(1:frames_min);
y_2_3 = y_2_3 - mean(y_2_3);
x_3_3 = x_3_3(1:frames_min);
x_3_3 = x_3_3 - mean(x_3_3);
y_3_3 = y_3_3(1:frames_min);
y_3_3 = y_3_3 - mean(y_3_3);
XY_total = [x_1_3; y_1_3; x_2_3; y_2_3; x_3_3; y_3_3];

figure(1)  % Plot the Raw Data Positional Data of Paint Can
subplot(3,1,1)
plot(1:frames_min, XY_total(2,:),1:frames_min, XY_total(1,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 3-Horizontal Displacement Case: Raw Positional Data of Paint Can, Cam 1");
legend("Y", "X")
subplot(3,1,2)
plot(1:frames_min, XY_total(4,:),1:frames_min, XY_total(3,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 3-Horizontal Displacement Case: Raw Positional Data of Paint Can, Cam 2");
legend("Y", "X")
subplot(3,1,3)
plot(1:frames_min, XY_total(6,:),1:frames_min, XY_total(5,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 3-Horizontal Displacement Case: Raw Positional Data of Paint Can, Cam 3")
legend("Y", "X")

% PCA on test 3
[U,S,V] = svd(XY_total/sqrt(frames_min-1), 'econ');
sig = diag(S);  % Calculating the energy of the truncations
figure(2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2)
title("Test 3-Horizontal Displacement Case: Energy Captured by Each Nonzero Singular Value");
xlabel("Singular Values"); ylabel("Energy");
axis([0 6 0 1])
set(gca,'Fontsize',14,'Xtick',0:1:6)
% Create Projection on first three Principal Components
figure(3)
subplot(2,1,1)
XY_proj = V';
plot(1:frames_min, XY_proj(1:2,:),'Linewidth',1.5)
title("Test 3-Horizontal Displacement Case: Movements on Principal Components");
ylabel("Displacement"); xlabel("Time (frames)"); 
legend("1st Principal Component", "2nd Principal Component",'location','southeast');
set(gca,'Fontsize',14)
subplot(2,1,2)
plot(1:frames_min, XY_proj(3:4,:),'Linewidth',1.5)
title("Test 3-Horizontal Displacement Case: Movements on Principal Components");
ylabel("Displacement"); xlabel("Time (frames)"); 
legend("3rd Principal Component","4th Principal Component",'location','southeast');
set(gca,'Fontsize',14)
%% Test 4: Horizontal Displacement and Rotation.
clear all; close all; clc;
load cam1_4.mat;  % -- Load Movie cam1_4 (Horizontal Displacement and Rotation from Cam 1)
[height1_4 width1_4 rgb num_frames1_4] = size(vidFrames1_4);
x_1_4 = [];
y_1_4 = [];
for j = 1:num_frames1_4  % -- Watch Movie
X = rgb2gray(vidFrames1_4(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X1 = double(X);  % converted to double precision for mathematical processing
X1(:,1:320) = 0;  % crop the video to the target region 
X1(:,460:end) = 0;
X1(1:225,:) = 0;
X1(400:end,:) = 0;
[M,I] = max(X1(:));
[y,x] = ind2sub(size(X1),I);
x_1_4 = [x_1_4 x];  % record the bright spot position in each frame
y_1_4 = [y_1_4 y];
end
load cam2_4.mat;  % -- Load Movie cam2_4 (Horizontal Displacement and Rotation from Cam 2)
[height2_4 width2_4 rgb num_frames2_4] = size(vidFrames2_4);
x_2_4 = [];
y_2_4 = [];
for j=1:num_frames2_4  % -- Watch Movie
X = rgb2gray(vidFrames2_4(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X2 = double(X);  % converted to double precision for mathematical processing
X(:,1:220) = 0;  % crop the video to the target region 
X(:,400:end) = 0;
X(1:80,:) = 0;
X(380:end,:) = 0;
[M,I] = max(X2(:));
[y,x] = ind2sub(size(X2),I);
x_2_4 = [x_2_4 x];  % record the bright spot position in each frame
y_2_4 = [y_2_4 y];
end
load cam3_4.mat;  % -- Load Movie cam3_4 (Horizontal Displacement and Rotation from Cam 3)
[height3_4 width3_4 rgb num_frames3_4] = size(vidFrames3_4);
x_3_4 = [];
y_3_4 = [];
for j=1:num_frames3_4  % -- Watch Movie
X = rgb2gray(vidFrames3_4(:,:,:,j));  % convert to grey scale 
% imshow(X); drawnow
X3 = double(X);  % converted to double precision for mathematical processing
X3(:,1:300) = 0;  % crop the video to the target region 
X3(:,480:end) = 0;
X3(1:140,:) = 0;
X3(280:end,:) = 0;
[M,I] = max(X3(:));
[y,x] = ind2sub(size(X3),I);
x_3_4 = [x_3_4 x];  % record the bright spot position in each frame
y_3_4 = [y_3_4 y];
end
% make all 3 cams start at same relative position
[Min,Ind] = min(y_1_4(1:25));
x_1_4 = x_1_4(Ind:end);
y_1_4 = y_1_4(Ind:end);
[Min,Ind] = min(y_2_4(1:25));
x_2_4 = x_2_4(Ind:end);
y_2_4 = y_2_4(Ind:end);
[Min,Ind] = min(x_3_4(1:25));
x_3_4 = x_3_4(Ind:end);
y_3_4 = y_3_4(Ind:end);

% trim each movie clip down to the same size
frames_min = min([length(x_1_4),length(x_2_4),length(x_3_4)]);
x_1_4 = x_1_4(1:frames_min);
x_1_4 = x_1_4 - mean(x_1_4);
y_1_4 = y_1_4(1:frames_min);
y_1_4 = y_1_4 - mean(y_1_4);
x_2_4 = x_2_4(1:frames_min);
x_2_4 = x_2_4 - mean(x_2_4);
y_2_4 = y_2_4(1:frames_min);
y_2_4 = y_2_4 - mean(y_2_4);
x_3_4 = x_3_4(1:frames_min);
x_3_4 = x_3_4 - mean(x_3_4);
y_3_4 = y_3_4(1:frames_min);
y_3_4 = y_3_4 - mean(y_3_4);
% Deal with deal with outliers in cam2 -the frames bright light not appearing in some frames
for i  = 2:frames_min
    if abs(x_2_4(1,i) - x_2_4(1,i-1)) > 50
        x_2_4(1,i) = x_2_4(1,i-1);
        y_2_4(1,i) = y_2_4(1,i-1);
    end
end
XY_total = [x_1_4; y_1_4; x_2_4; y_2_4; x_3_4; y_3_4];
figure(1)  % Plot the Raw Data Positional Data of Paint Can
subplot(3,1,1)
plot(1:frames_min, XY_total(2,:),1:frames_min, XY_total(1,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 4-Horizontal Displacement and Rotation: Raw Positional Data of Paint Can, Cam 1");
legend("Y", "X")
subplot(3,1,2)
plot(1:frames_min, XY_total(4,:),1:frames_min, XY_total(3,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 4-Horizontal Displacement and Rotation: Raw Positional Data of Paint Can, Cam 2");
legend("Y", "X")
subplot(3,1,3)
plot(1:frames_min, XY_total(6,:),1:frames_min, XY_total(5,:), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Test 4-Horizontal Displacement and Rotation: Raw Positional Data of Paint Can, Cam 3")
legend("Y", "X")

% PCA on test 3
[U,S,V] = svd(XY_total/sqrt(frames_min-1), 'econ');
sig = diag(S);  % Calculating the energy of the truncations
figure(2)
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2)
title("Test 4-Horizontal Displacement and Rotation: Energy Captured by Each Nonzero Singular Value");
xlabel("Singular Values"); ylabel("Energy");
axis([0 6 0 1])
set(gca,'Fontsize',14,'Xtick',0:1:6)
% Create Projection on first three Principal Components
figure(3)
subplot(2,1,1)
XY_proj = V';
plot(1:frames_min, XY_proj(1:2,:),'Linewidth',1.5)
title("Test 4-Horizontal Displacement and Rotation: Movements on Principal Components");
ylabel("Displacement"); xlabel("Time (frames)"); 
legend("1st Principal Component", "2nd Principal Component", ... 
       "3rd Principal Component",'location','southeast');
set(gca,'Fontsize',14)
subplot(2,1,2)
plot(1:frames_min, XY_proj(3:4,:),'Linewidth',1.5)
title("Test 4-Horizontal Displacement and Rotation: Movements on Principal Components");
ylabel("Displacement"); xlabel("Time (frames)"); 
legend("3rd Principal Component","4th Principal Component",'location','southeast');
set(gca,'Fontsize',14)