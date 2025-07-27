%clear;
load('bootstrap.mat');
%load('sidewalk.mat');
whos;
%X=T;
[m, n, p] = size(X); 
%disp(X);
X = double(X);
X_reshaped = reshape(X, [m*n, p])';  

Linear_Ti = ones(size(X_reshaped));
lambda = 1/sqrt(max(size(X_reshaped)));
X_reshaped=X_reshaped/256;
[L_ini, S_ini] = NoiseRobustPCA_ADMM_t1_nuclear(X_reshaped,Linear_Ti,0,0,m*n*p, 1e-6, 1e-9,1e-7,1e-7, 1e-2, 100,0,0,X_reshaped,0);
tic
[L, S] = NoiseRobustPCA_ADMM_t1_TL1(X_reshaped,Linear_Ti,1,10,m*n*p, 1e-4, 1e-9,1e-7,1e-7, 1e-2, 100,0,0,L_ini,S_ini);
%toc
elapsed_time = toc;
disp("time");
disp(elapsed_time);
%figure;
L=L*256;
S=S*256;
L_re=reshape(L', [m, n, p]);
S_re=reshape(S', [m, n, p]);

disp(['Max X: ', num2str(max(X(:)))]);
disp(['Max L: ', num2str(max(L(:)))]);
disp(['Max S: ', num2str(max(S(:)))]);

[height, width, numFrames] = size(X);

% save output video
%outputVideo = VideoWriter('combined_video_air_TL1.avi'); 
outputVideo = VideoWriter(fullfile('noisyRPCAexperiment', 'real data','combined_video_boot_nuclear_t.avi'));

outputVideo.FrameRate = 10; 
open(outputVideo);

for frame = 1:numFrames
    combinedFrame = zeros(height, width * 3+10);
    S_re=abs(S_re);
    combinedFrame(:, 1:width) = X(:, :, frame); 
    combinedFrame(:, width+1+5:width*2+5) = L_re(:, :, frame); 
    combinedFrame(:, width*2+1+10:end) = S_re(:, :, frame); 

    writeVideo(outputVideo, uint8(combinedFrame)); 
end

close(outputVideo); 