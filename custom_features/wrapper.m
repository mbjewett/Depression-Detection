%% Author: Soo Jin Park, Abeer Alwan
%% Modified by: Vijay Ravi, Jinhan Wang

%$ This Matlab wrapper file can be used for extraction of custom features and 
%% saving it into csv files. Please modify the file paths and feature extraction function before 
%% running this. 

clear;clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  Get Paths
feats_scp = '/media/amber/pikachu/vijay/workdir/model_arch/214b_project/EATD/labels/feat.scp';
fid = fopen(feats_scp);
M = textscan(fid, '%s%s', 'Delimiter','\t');
wav_files = M{1};
feat_files = M{2};
number_of_segments = length(wav_files);
if length(wav_files) ~= length(feat_files)
    disp("Error! Number of files mismatch");
    exit
end

%%
for cnt = 1:length(wav_files)
    
    sndFilePath = wav_files{cnt};
    featName = feat_files{cnt};
    fprintf('Processing file number %d: %s\n', cnt,sndFilePath);
    [ftr] = <custom feature extraction function>;
    end

    inds = strfind(featName,'/');
    dirstore = featName(1:inds(end)-1);
    
    if(~exist(dirstore))
        mkdir(dirstore);
    end
   
    dlmwrite(featName,ftr);
    
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%