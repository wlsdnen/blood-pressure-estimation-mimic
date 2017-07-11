% This program is intended to delineate the fiducial points of pulse waveforms
% Reference:
%   BN Li, MC Dong & MI Vai (2010)
%   On an automatic delineator for arterial blood pressure waveforms
%   Biomedical Signal Processing and Control 5(1) 76-81.

% LI Bing Nan @ University of Macau, Feb 2007
%   Revision 2.0.5, Apr 2009

% clear all;

% load Part_1      % http://www.physionet.org/pn3/mghdb/

% abpview(Part_1{1}(2, :),onsetp,peakp,dicron);
ppg     = {};
ecg 	= {};
bp      = {};
bp_label= [];
index   = [];
len     = [];

for i = 1:length(Part_1)
    
    [onsetp,peakp,dicron] = delineator(Part_1{i}(2, :), 125);
    idx = length(onsetp);
    %write files by cell element
    
    for j = 1:idx
        
        if j + 1 > idx
            break;
        else
            if length (Part_1{i}(1, onsetp(j)+1:onsetp(j+1))) > 70
                continue;
            elseif length (Part_1{i}(1, onsetp(j)+1:onsetp(j+1))) < 40
                continue;
%             elseif Part_1{i}(2, peakp(j)) < 110
%                 continue;
%             elseif Part_1{i}(2, onsetp(j)) > 90
%                 continue;
%             elseif Part_1{i}(2, onsetp(j)) < 60
%                 continue;
            else
                ppg{end+1}  = Part_1{i}(1, onsetp(j)+1:onsetp(j+1));
                ecg{end+1}  = Part_1{i}(3, onsetp(j)+1:onsetp(j+1));
                bp{end+1}   = Part_1{i}(2, onsetp(j)+1:onsetp(j+1));
                bp_label(end+1, 1:2)    = [Part_1{i}(2, peakp(j)), Part_1{i}(2, onsetp(j+1))];
                index(end+1, 1:2)       = [i, onsetp(j)];
                len(end+1)              = length(bp{end});
            end
        end
    end
end
