% TASK: Complete the code of the PMBM filter class, which contains necessary functions to implement a track-oriented Poisson multi-Bernoulli mixture (PMBM) filter:
%     1. Prediction of Bernoulli component.
%     2. Misdetection update of Bernoulli component.
%     3. Object detection update of Bernoulli component.
%     4. Prediction of Poisson Point Process (PPP).
%     5. Misdetection update of PPP.
%     6. Object detection update of PPP.
%     7. PMBM prediction.
%     8. PMBM update.
%     9. Object states extraction.
% 
% For task 4, i.e., prediction of PPP, your implementation should consist of the following steps:
%     1. Predict Poisson intensity for pre-existing objects.
%     2. Incorporate Poisson birth intensity into Poisson intensity for pre-existing objects.
% 
% For task 6, i.e., object detection update of PPP, your implementation should consist of the following steps:
%     1. For each mixture component in the PPP intensity, perform Kalman update and calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
%     2. Perform Gaussian moment matching for the updated object state densities resulted from being updated by the same detection.
%     3. The returned likelihood should be the sum of the predicted likelihoods calculated for each mixture component in the PPP intensity and the clutter intensity. (You can make use of the normalizeLogWeights function to achieve this.)
%     4. The returned existence probability of the Bernoulli component is the ratio between the sum of the predicted likelihoods and the returned likelihood. (Be careful that the returned existence probability is in decimal scale while the likelihoods you calculated beforehand are in logarithmic scale.)
% 
% For task 8, i.e., PMBM update, your implementation should consist of the following steps:
%     1. Perform ellipsoidal gating for each Bernoulli state density and each mixture component in the PPP intensity.
%     2. Bernoulli update. For each Bernoulli state density, create a misdetection hypothesis (Bernoulli component), and m object detection hypothesis (Bernoulli component), where m is the number of detections inside the ellipsoidal gate of the given state density.
%     3. Update PPP with detections. Note that for detections that are not inside the gate of undetected objects, create dummy Bernoulli components with existence probability r = 0; in this case, the corresponding likelihood is simply the clutter intensity.
%     4. For each global hypothesis, construct the corresponding cost matrix and use Murty's algorithm to obtain the M best global hypothesis with highest weights. Note that for detections that are only inside the gate of undetected objects, they do not need to be taken into account when forming the cost matrix. 
%     5. Update PPP intensity with misdetection.
%     6. Update the global hypothesis look-up table.
%     7. Prune global hypotheses with small weights and cap the number. 
%     8. Prune local hypotheses (or hypothesis trees) that do not appear in the maintained global hypotheses, and re-index the global hypothesis look-up table.
% 
% 
% For task 9, i.e., object states extraction, your implementation should consist of the following steps:
%     1. Find the multi-Bernoulli with the highest weight.
%     2. Extract the mean of the object state density from Bernoulli components with probability of existence no less than a threshold.
% 
% The M-best 2D assignment solver has been provided as a reference function.
%     [col4rowBest,row4colBest,gainBest]=kBest2DAssign(C,k)
% 
% KBEST2DASSIGN: Find the k lowest cost 2D assignments for the two-dimensional assignment problem with a rectangular cost matrix C.
% 
% INPUT:      C: A numRowXnumCol cost matrix.
% OUTPUTS:    col4rowBest: A numRowXk vector where the entry in each element is an assignment of the element in that row to a column. 0 entries signify unassigned rows.
%             row4colbest: A numColXk vector where the entry in each element is an assignment of the element in that column to a row. 0 entries signify unassigned columns.
%             gainBest: A kX1 vector containing the sum of the values of the assigned elements in C for all of the hypotheses.
% Note:
%     It is assumed that the object survival probability P_S is constant.
%     We can use gating to further group objects into subsets and process each subset indepedently. However, this is NOT implemented in this task.
%     When normalising or summing weights in logarithmic scale, you can call function normalizeLogWeights, which has also been provided as a reference function. 
%     When constructing the cost matrix, if measurement j does not fall inside the gate of object i, set the corresponding entry  to .
%     If the constructed cost matrix is empty, do not forget to consider the case that all the detected objects are missed detected.
%     Set parameter k used in kBest2DAssign to ceil(*obj.reduction.M), where  denotes the weight of global hypothesis h and it satisfies that .
%     The hypothesis look-up table maintained in the track-oriented PMBM filter is similar to the one maintained in the track-oriented MHT. The difference is that the table in the PMBM filter can have entries with zero value.
%     Always append new hypothesis tree to the right side of the existing hypothesis trees. The same applies when you expand the hypothesis look-up table. This is in consistent with the video content and important for you to pass the tests.
%     When pruning local/global hypotheses, make sure that the number of rows of the global hypothesis table always matches the length of the global hypothesis weight vector, and that the number of columns of the global hypothesis table always matches the number of local hypothesis trees.
% 
% Example: re-index hypothesis look-up table.
% 
% Suppose the hypothesis look-up table before re-indexing is . Then after re-indexing, the hypothesis look-up table may look like .
% 
% Hint:
%     If you want to apply a function to each element of a struct/cell array, you can use MATLAB command arrayfun/cellfun, which makes your implementation faster than using for loops. 
%     When pruning low weight data association events, you can call the hypothesisReduction.prune method you have written in the first home assignment. You simply need to change the second input parameter from struct array to hypotheses indices. Similar trick also applies to hypothesisReduction.cap.
%     When re-indexing the look-up table, you might find MATLAB function unique useful.
%     For the maintainance of the hypothesis look-up table, you may take a look at the provided recycling function.
% Files referenced:
%     kBest2DAssign.m
%     modelgen.m
%     measmodel.m
%     GaussianDensity.m
%     hypothesisReduction.m
%     motionmodel.m
%     normalizeLogWeights.m
%     log_mvnpdf.m
% Note that obj.density is a function handle bound to MATLAB class GaussianDensity. For instance, you can simply call obj.density.update to perform a Kalman update instead of using your own code. 



classdef PMBMfilter
    %PMBMFILTER is a class containing necessary functions to implement the
    %PMBM filter
    %Model structures need to be called:
    %    sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    properties
        density %density class handle
        paras   %%parameters specify a PMBM
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PMBMfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture)
            %       of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.PPP.w: weights of mixture components in PPP
            %       intensity --- vector of size (number of mixture
            %       components x 1) in logarithmic scale
            %       obj.paras.PPP.states: parameters of mixture components
            %       in PPP intensity struct array of size (number of
            %       mixture components x 1)
            %       obj.paras.MBM.w: weights of MBs --- vector of size
            %       (number of MBs (global hypotheses) x 1) in logarithmic 
            %       scale
            %       obj.paras.MBM.ht: hypothesis table --- matrix of size
            %       (number of global hypotheses x number of hypothesis
            %       trees). Entry (h,i) indicates that the (h,i)th local
            %       hypothesis in the ith hypothesis tree is included in
            %       the hth global hypothesis. If entry (h,i) is zero, then
            %       no local hypothesis from the ith hypothesis tree is
            %       included in the hth global hypothesis.
            %       obj.paras.MBM.tt: local hypotheses --- cell of size
            %       (number of hypothesis trees x 1). The ith cell contains
            %       local hypotheses in struct form of size (number of
            %       local hypotheses in the ith hypothesis tree x 1). Each
            %       struct has two fields: r: probability of existence;
            %       state: parameters specifying the object density
            
            obj.density = density_class_handle;
            obj.paras.PPP.w = [birthmodel.w]';
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = {};
        end
        
        function Bern = Bern_predict(obj,Bern,motionmodel,P_S)
            %BERN_PREDICT performs prediction step for a Bernoulli component
            %INPUT: Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar;
            %                          state: a struct contains parameters
            %                          describing the object pdf
            %       P_S: object survival probability
            state = obj.density.predict(Bern.state, motionmodel);
            r     = P_S * Bern.r;
            
            Bern.state = state;
            Bern.r     = r;

        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D)
            %BERN_UNDETECTED_UPDATE calculates the likelihood of missed
            %detection, and creates new local hypotheses due to missed
            %detection.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth local
            %       hypothesis in the ith hypothesis tree. 
            %       P_D: object detection probability --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %       with fields: r: probability of existence --- scalar;
            %                    state: a struct contains parameters
            %                    describing the object pdf
            %       lik_undetected: missed detection likelihood --- scalar
            %       in logorithmic scale
            hypothesis = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            P_MD = 1 - P_D;
            Bern.r = (hypothesis.r * P_MD)/(1 - hypothesis.r + hypothesis.r * P_MD);
            Bern.state = hypothesis.state;   % no update 
            lik_undetected = log(1 - hypothesis.r + hypothesis.r * P_MD);
            assert(isreal(lik_undetected), 'Not a float');
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            %BERN_DETECTED_UPDATE_LIK calculates the predicted likelihood
            %for a given local hypothesis. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth
            %       local hypothesis in the ith hypothesis tree.
            %       z: measurement array --- (measurement dimension x
            %       number of measurements)
            %       P_D: object detection probability --- scalar
            %OUTPUT:lik_detected: predicted likelihood --- (number of
            %measurements x 1) array in logarithmic scale 
            hypothesis = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            lik_detected = log(hypothesis.r) + log(P_D) + obj.density.predictedLikelihood(hypothesis.state, z, measmodel);
            assert(isequal(size(lik_detected), [size(z, 2), 1]), 'implement error');
            assert(isreal(lik_detected), 'Not a float');
        end
        
        function Bern = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            %BERN_DETECTED_UPDATE_STATE creates the new local hypothesis
            %due to measurement update. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %                 local hypotheses. (i,j) indicates the jth
            %                 local hypothesis in the ith hypothesis tree.
            %       z: measurement vector --- (measurement dimension x 1)
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar; 
            %                          state: a struct contains parameters
            %                          describing the object pdf 
            hypothesis = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            Bern.r = 1;
            Bern.state = obj.density.update(hypothesis.state, z, measmodel);   % no update 
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            %PPP_PREDICT performs predicion step for PPP components
            %hypothesising undetected objects.
            %INPUT: P_S: object survival probability --- scalar          

            num_of_components = length(obj.paras.PPP.w);
            % predict parameters from previous time step 
            for i = 1 : length(obj.paras.PPP.w)
                obj.paras.PPP.w(i) = obj.paras.PPP.w(i) + log(P_S);
                obj.paras.PPP.states(i) = obj.density.predict(obj.paras.PPP.states(i), motionmodel);
            end
            
            % add new components
            for i = 1 : length(birthmodel)
                obj.paras.PPP.w = [obj.paras.PPP.w; birthmodel(i).w];
                birth_state = struct('x', birthmodel(i).x ,'P', birthmodel(i).P);
                obj.paras.PPP.states = [obj.paras.PPP.states; birth_state];
            end

            assert(isequal(size(obj.paras.PPP.w), [num_of_components + length(birthmodel), 1]), 'implement error');
            assert(isequal(size(obj.paras.PPP.states), [num_of_components + length(birthmodel), 1]), 'implement error');

        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,indices,z,measmodel,P_D,clutter_intensity)
            %PPP_DETECTED_UPDATE creates a new local hypothesis by
            %updating the PPP with measurement and calculates the
            %corresponding likelihood.
            %INPUT: z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %       clutter_intensity: Poisson clutter intensity --- scalar
            %       indices: boolean vector, if measurement z is inside the
            %       gate of mixture component i, then indices(i) = true
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %             scalar;
            %             state: a struct contains parameters describing
            %             the object pdf
            %       lik_new: predicted likelihood of PPP --- scalar in
            %       logarithmic scale 
            updated_states = []; % struct array of size (# ? , 1)
            w = []; % struct array of size (# ?, 1)
            for i = 1 : length(indices)
                is_ingate = indices(i);
                if is_ingate
                    updated_states = [updated_states; obj.density.update(obj.paras.PPP.states(i), z, measmodel)];
                    w = [w; obj.paras.PPP.w(i) + log(P_D) + obj.density.predictedLikelihood(obj.paras.PPP.states(i), z, measmodel)]; %logarithm domain 
                end
            end
            [normalized_w, ~] = normalizeLogWeights(w);
            mixed_state = obj.density.momentMatching(normalized_w, updated_states);
            r = sum(exp(w)) / (clutter_intensity + sum(exp(w)));
            Bern = struct('r', r, 'state', mixed_state);
            lik_new = log(clutter_intensity + sum(exp(w)));
            assert(isreal(lik_new), 'Not a float');
        end
        
        function obj = PPP_undetected_update(obj,P_D)
            %PPP_UNDETECTED_UPDATE performs PPP update for missed detection.
            %INPUT: P_D: object detection probability --- scalar
            for i = 1 : length(obj.paras.PPP.w)
                obj.paras.PPP.w(i) = obj.paras.PPP.w(i) + log(1 - P_D);
            end        
        end
        
        function obj = PPP_reduction(obj,prune_threshold,merging_threshold)
            %PPP_REDUCTION truncates mixture components in the PPP
            %intensity by pruning and merging
            %INPUT: prune_threshold: pruning threshold --- scalar in
            %       logarithmic scale
            %       merging_threshold: merging threshold --- scalar
            [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.prune(obj.paras.PPP.w, obj.paras.PPP.states, prune_threshold);
            if ~isempty(obj.paras.PPP.w)
                [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.merge(obj.paras.PPP.w, obj.paras.PPP.states, merging_threshold, obj.density);
            end
        end
        
        function obj = Bern_recycle(obj,prune_threshold,recycle_threshold)
            %BERN_RECYCLE recycles Bernoulli components with small
            %probability of existence, adds them to the PPP component, and
            %re-index the hypothesis table. If a hypothesis tree contains no
            %local hypothesis after pruning, this tree is removed. After
            %recycling, merge similar Gaussian components in the PPP
            %intensity
            %INPUT: prune_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold are pruned ---
            %       scalar
            %       recycle_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold needed to be
            %       recycled --- scalar
            
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold & x.r>=prune_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Here, we should also consider the weights of different MBs
                    idx_t = find(idx);
                    n_h = length(idx_t);
                    w_h = zeros(n_h,1);
                    for j = 1:n_h
                        idx_h = obj.paras.MBM.ht(:,i) == idx_t(j);
                        [~,w_h(j)] = normalizeLogWeights(obj.paras.MBM.w(idx_h));
                    end
                    %Recycle
                    temp = obj.paras.MBM.tt{i}(idx);
                    obj.paras.PPP.w = [obj.paras.PPP.w;log([temp.r]')+w_h];
                    obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                end
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Remove Bernoullis
                    obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                    %Update hypothesis table, if a Bernoulli component is
                    %pruned, set its corresponding entry to zero
                    idx = find(idx);
                    for j = 1:length(idx)
                        temp = obj.paras.MBM.ht(:,i);
                        temp(temp==idx(j)) = 0;
                        obj.paras.MBM.ht(:,i) = temp;
                    end
                end
            end
            
            %Remove tracks that contains no valid local hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                %Ensure the algorithm still works when all Bernoullis are
                %recycled
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i) > 0;
                [~,~,obj.paras.MBM.ht(idx,i)] = unique(obj.paras.MBM.ht(idx,i),'rows','stable');
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows','stable');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
            
        end
        
        function obj = PMBM_predict(obj,P_S,motionmodel,birthmodel)
            %PMBM_PREDICT performs PMBM prediction step.
            obj = obj.PPP_predict(motionmodel, birthmodel, P_S);
            for i = 1 : length(obj.paras.MBM.tt)
                obj.paras.MBM.tt{i} = arrayfun(@(x) obj.Bern_predict(x, motionmodel, P_S), obj.paras.MBM.tt{i});
            end
        end
        
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,w_min,M)
            %PMBM_UPDATE performs PMBM update step.
            %INPUT: z: measurements --- array of size (measurement
            %       dimension x number of measurements)
            %       gating: a struct with two fields that specifies gating
            %       parameters: P_G: gating size in decimal --- scalar;
            %                   size: gating size --- scalar.
            %       wmin: hypothesis weight pruning threshold --- scalar in
            %       logarithmic scale
            %       M: maximum global hypotheses kept
            m = size(z,2);                      %number of measurements received
            used_meas_u = false(m,1);           %measurement indices inside the gate of undetected objects
            nu = length(obj.paras.PPP.states);  %number of mixture components in PPP intensity
            gating_matrix_u = false(m,nu);
            for i = 1:nu
                %Perform gating for each mixture component in the PPP intensity
                [~,gating_matrix_u(:,i)] = ...
                    obj.density.ellipsoidalGating(obj.paras.PPP.states(i),z,measmodel,gating.size);
                used_meas_u = used_meas_u | gating_matrix_u(:,i);
            end
            
            n_tt = length(obj.paras.MBM.tt);    %number of pre-existing hypothesis trees
            likTable = cell(n_tt,1);            %initialise likelihood table, one for each hypothesis tree
            gating_matrix_d = cell(n_tt,1);
            used_meas_d = false(m,1);           %measurement indices inside the gate of detected objects
            for i = 1:n_tt
                %number of local hypotheses in hypothesis tree i
                num_hypo = length(obj.paras.MBM.tt{i});
                %construct gating matrix
                gating_matrix_d{i} = false(m,num_hypo);
                for j = 1:num_hypo
                    %Perform gating for each local hypothesis
                    [~,gating_matrix_d{i}(:,j)] = obj.density.ellipsoidalGating(obj.paras.MBM.tt{i}(j).state,z,measmodel,gating.size);
                    used_meas_d = used_meas_d | gating_matrix_d{i}(:,j);
                end
            end
            
            %measurement indices inside the gate
            used_meas = used_meas_d | used_meas_u;
            %find indices of measurements inside the gate of undetected
            %objects but not detected objects
            used_meas_u_not_d = used_meas > used_meas_d;
            
            %Update detected objects
            %obtain measurements that are inside the gate of detected objects
            z_d = z(:,used_meas_d);
            m = size(z_d,2);
            gating_matrix_d = cellfun(@(x) x(used_meas_d,:), gating_matrix_d, 'UniformOutput',false);
            n_tt_upd = n_tt + m;                %number of hypothesis trees
            hypoTable = cell(n_tt_upd,1);       %initialise hypothesis table, one for each hypothesis tree
            for i = 1:n_tt
                %number of local hypotheses in hypothesis tree i
                num_hypo = length(obj.paras.MBM.tt{i});
                %initialise likelihood table for hypothesis tree i
                likTable{i} = -inf(num_hypo,m+1);
                %initialise hypothesis table for hypothesis tree i
                hypoTable{i} = cell(num_hypo*(m+1),1);
                for j = 1:num_hypo
                    %Missed detection
                    [hypoTable{i}{(j-1)*(m+1)+1},likTable{i}(j,1)] = Bern_undetected_update(obj,[i,j],sensormodel.P_D);
                    %Update with measurement
                    likTable{i}(j,[false;gating_matrix_d{i}(:,j)]) = ...
                        Bern_detected_update_lik(obj,[i,j],z_d(:,gating_matrix_d{i}(:,j)),measmodel,sensormodel.P_D);
                    for jj = 1:m
                        if gating_matrix_d{i}(jj,j)
                            hypoTable{i}{(j-1)*(m+1)+jj+1} = Bern_detected_update_state(obj,[i,j],z_d(:,jj),measmodel);
                        end
                    end
                end
            end
            
            %Update undetected objects
            lik_new = -inf(m,1);
            gating_matrix_ud = gating_matrix_u(used_meas_d,:);
            %Create new hypothesis trees, one for each measurement inside
            %the gate 
            for i = 1:m
                if any(gating_matrix_ud(i,:))
                    [hypoTable{n_tt+i,1}{1}, lik_new(i)] = ...
                        PPP_detected_update(obj,gating_matrix_ud(i,:),z_d(:,i),measmodel,sensormodel.P_D,sensormodel.intensity_c);
                else
                    %For measurements not inside the gate of undetected
                    %objects, set likelihood to clutter intensity
                    lik_new(i) = log(sensormodel.intensity_c);
                end
            end
            used_meas_ud = sum(gating_matrix_ud, 2) >= 1; 
            
            %Cost matrix for first detection of undetected objects
            L2 = inf(m);
            L2(logical(eye(m))) = -lik_new;
            
            %Update global hypothesis
            w_upd = [];             
            ht_upd = zeros(0,n_tt_upd);
            H_upd = 0;
            
            %Number of global hypothesis
            H = length(obj.paras.MBM.w);
            if H == 0 %if there is no pre-existing hypothesis tree
                w_upd = 0;
                H_upd = 1;
                ht_upd = zeros(1,m);
                ht_upd(used_meas_ud) = 1;
            else
                for h = 1:H
                    %Cost matrix for detected objects
                    L1 = inf(m,n_tt);
                    lik_temp = 0;
                    for i = 1:n_tt
                        hypo_idx = obj.paras.MBM.ht(h,i);
                        if hypo_idx~=0
                            L1(:,i) = -(likTable{i}(hypo_idx,2:end) - likTable{i}(hypo_idx,1));
                            %we need add the removed weights back to
                            %calculate the updated global hypothesis weight
                            lik_temp = lik_temp + likTable{i}(hypo_idx,1);
                        end
                    end
                    %Cost matrix of size m-by-(n+m)
                    L = [L1 L2];
                    
                    if isempty(L)
                        %Consider the case that no measurements are inside
                        %the gate, thus missed detection
                        gainBest = 0;
                        col4rowBest = 0;
                    else
                        %Obtain M best assignments using Murty's algorithm
                        [col4rowBest,~,gainBest] = kBest2DAssign(L,ceil(exp(obj.paras.MBM.w(h)+log(M))));
                        %Obtain M best assignments using Gibbs sampling
%                       [col4rowBest,gainBest] = assign2DByGibbs(L,100,ceil(exp(obj.paras.MBM.w(h)+log(M))));
                    end
                    
                    %Restore weights
                    w_upd = [w_upd;-gainBest+lik_temp+obj.paras.MBM.w(h)];
                    
                    %Update global hypothesis look-up table
                    Mh = length(gainBest);
                    ht_upd_h = zeros(Mh,n_tt_upd);
                    for j = 1:Mh
                        ht_upd_h(j,1:n_tt_upd) = 0;
                        for i = 1:n_tt
                            if obj.paras.MBM.ht(h,i) ~= 0
                                idx = find(col4rowBest(:,j)==i, 1);
                                if isempty(idx)
                                    %missed detection
                                    ht_upd_h(j,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+1;
                                else
                                    %measurement update
                                    ht_upd_h(j,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+idx+1;
                                end
                            end
                        end
                        for i = n_tt+1:n_tt_upd
                            idx = find(col4rowBest(:,j)==i, 1);
                            if ~isempty(idx) && used_meas_ud(idx)
                                %measurement update for PPP
                                ht_upd_h(j,i) = 1;
                            end
                        end
                    end
                    H_upd = H_upd + Mh;
                    ht_upd = [ht_upd;ht_upd_h];
                end
                
                %Normalize global hypothesis weights
                 w_upd = normalizeLogWeights(w_upd);
                
            end
            
            %Append new hypothesis trees that created by measurements
            %inside the gate of undetected objects but not detected objects
            z_u_not_d = z(:,used_meas_u_not_d);
            num_u_not_d = size(z_u_not_d,2);
            gating_matrix_u_not_d = gating_matrix_u(used_meas_u_not_d,:);
            for i = 1:num_u_not_d
                [hypoTable{n_tt_upd+i,1}{1}, ~] = ...
                    PPP_detected_update(obj,gating_matrix_u_not_d(i,:),z_u_not_d(:,i),measmodel,sensormodel.P_D,sensormodel.intensity_c);
            end
            ht_upd = [ht_upd ones(H_upd,num_u_not_d)];
            
            %Update undetected objects with missed detection
            obj = PPP_undetected_update(obj,sensormodel.P_D);
            
            %Prune hypotheses with weight smaller than the specified
            %threshold 
            [w_upd, hypo_idx] = hypothesisReduction.prune(w_upd,1:H_upd,w_min);
            ht_upd = ht_upd(hypo_idx,:);
            w_upd = normalizeLogWeights(w_upd);
            
            %Keep at most M hypotheses with the highest weights
            [w_upd, hypo_idx] = hypothesisReduction.cap(w_upd,1:length(w_upd),M);
            ht_upd = ht_upd(hypo_idx,:);
            obj.paras.MBM.w = normalizeLogWeights(w_upd);
            
            %Remove empty hypothesis trees
            if ~isempty(ht_upd)
                idx = sum(ht_upd,1) >= 1;
                ht_upd = ht_upd(:,idx);
                hypoTable = hypoTable(idx);
                n_tt_upd = size(ht_upd,2);
            end
            
            %Prune local hypotheses that do not appear in maintained global
            %hypotheses 
            obj.paras.MBM.tt = cell(n_tt_upd,1);
            for i = 1:n_tt_upd
                temp = ht_upd(:,i);
                hypoTableTemp = hypoTable{i}(unique(temp(temp~=0), 'stable'));
                obj.paras.MBM.tt{i} = [hypoTableTemp{:}]';
            end
            
            %Re-index hypothesis table
            for i = 1:n_tt_upd
                idx = ht_upd(:,i) > 0;
                [~,~,ht_upd(idx,i)] = unique(ht_upd(idx,i),'rows','stable');
            end
            
            obj.paras.MBM.ht = ht_upd;
            
        end
        
        function estimates = PMBM_estimator(obj,threshold)
            %PMBM_ESTIMATOR performs object state estimation in the PMBM
            %filter
            %INPUT: threshold (if exist): object states are extracted from
            %       Bernoulli components with probability of existence no
            %       less than this threshold in Estimator 1. Given the
            %       probabilities of detection and survival, this threshold
            %       determines the number of consecutive misdetections
            %OUTPUT:estimates: estimated object states in matrix form of
            %       size (object state dimension) x (number of objects)
            %%%
            %First, select the multi-Bernoulli with the highest weight.
            %Second, report the mean of the Bernoulli components whose
            %existence probability is above a threshold. 
            [~, I] = max(obj.paras.MBM.w);
            indices_of_localhypothesis = obj.paras.MBM.ht(I, :);
            estimates = [];
            for i = 1 : length(indices_of_localhypothesis)
                idx = indices_of_localhypothesis(i); % local hypothesis
                if (idx) ~= 0 && (obj.paras.MBM.tt{i}(idx).r > threshold)
                    estimates = [estimates obj.paras.MBM.tt{i}(idx).state.x];
                end
            end
        end
 
    end
end