%% MATLAB Function: RLS algorithm for real-valued TRDFE used with
%% PAM transmission.
%% ===================================================================== %
%% Author: Isaac N. O. Osahon
%% MATLAB version R2020a

%========== Parameters========================%
%  rxeq: Equalised output
%  rxd: Estimated input constellation to the equalised output  
%  weightVec: Weight vector of inputs
%  tcmeq: Time taken to execute the algorithm
%  nfTaps: Number of Forward Taps
%  nbTaps: Number of FeedBack Taps
%  eqSymDel: symbol shift between equalizer input and output
%  ffInput: input signal to the equalizer
%  desired: training symbols
%  const: constellation
%  lamda: Forgetting factor (0 << lamda <= 1)
%  FSE: Fractionally Spaced-Symbol Equalization if FSE >=1
%  ddstr: adapt after training if this variable is anything but 'Ndd'

function [rxeq,rxd,weightVec,tcmeq] = myRLSEQ(nfTaps,nbTaps,...
    eqSymDel,ffInput,desired,const,lamda,FSE,ddstr)
%===================== Initialization=====================%
n_train = length(desired);
n_data = length(ffInput)/FSE;
nTaps = nfTaps+nbTaps+1;
weightVec = zeros(nTaps,1); 
SdMat = eye(nTaps)*n_train/sum((ffInput(1:n_train)).^2);
rxeq = zeros(1,n_data);
ffInput = ffInput(:);
if nfTaps>FSE
    ffInput = [zeros(nfTaps-FSE,1);ffInput];
end
desired = [zeros(eqSymDel,1);desired(:);zeros(n_data-n_train-eqSymDel,1)];
if nbTaps >= 1
    fbinput = [zeros(nbTaps,1);desired];
else
    fbinput = [];
end
part = 0.5*(const(1:end-1) + const(2:end));

%======================= Iteration ==========================%
tcmeq = cputime;
for cct = 1:n_data
    cctFSE = FSE*(cct-1)+1;
    inputVec = [1;ffInput(cctFSE:cctFSE+nfTaps-1);fbinput(cct:cct+nbTaps-1)];
    ycct = weightVec.'*inputVec;
    
    if cct>n_train + eqSymDel
        [~,desired(cct)] = quantiz(ycct,part,const);
        if nbTaps >= 1
            fbinput(cct+nbTaps) = desired(cct);
        end
        if strcmpi(ddstr,'Ndd')
            rxeq(cct) = ycct;
        end
    end
    if ~strcmpi(ddstr,'Ndd') || cct<=n_train + eqSymDel
        errcct = desired(cct)-ycct;
        phi = SdMat*inputVec;
        SdMat = (SdMat - (phi*phi.'/(lamda + phi.'*inputVec)))/lamda;
        weightVec = weightVec + errcct*SdMat*inputVec;
        rxeq(cct) = weightVec.'*inputVec;
    end
    if cct > eqSymDel
        [~,desired(cct)] = quantiz(rxeq(cct),part,const);
    end
end
tcmeq = cputime - tcmeq;
rxd = desired;
rxeq = rxeq(:);
end

