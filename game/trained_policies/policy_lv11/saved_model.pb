
й
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Э
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
І
%QNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*6
shared_name'%QNetwork/EncodingNetwork/dense/kernel

9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes

: d*
dtype0

#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#QNetwork/EncodingNetwork/dense/bias

7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes
:d*
dtype0

QNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d0*(
shared_nameQNetwork/dense_1/kernel

+QNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/kernel*
_output_shapes

:d0*
dtype0

QNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameQNetwork/dense_1/bias
{
)QNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/bias*
_output_shapes
:0*
dtype0

NoOpNoOp
Э
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueўBћ Bє
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
	3


0
 
ge
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#QNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEQNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEQNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
t
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
n
_postprocessing_layers
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
	3

0
1
2
	3
­
layer_metrics
metrics

layers
layer_regularization_losses
 non_trainable_variables
regularization_losses
trainable_variables
	variables

!0
"1
 

0
1

0
1
­
#layer_metrics
$metrics

%layers
&layer_regularization_losses
'non_trainable_variables
regularization_losses
trainable_variables
	variables
 

0
	1

0
	1
­
(layer_metrics
)metrics

*layers
+layer_regularization_losses
,non_trainable_variables
regularization_losses
trainable_variables
	variables
 
 

0
1
 
 
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
 
 

!0
"1
 
 
 
 
 
 
 
 
 
 
­
5layer_metrics
6metrics

7layers
8layer_regularization_losses
9non_trainable_variables
-regularization_losses
.trainable_variables
/	variables
 

0
1

0
1
­
:layer_metrics
;metrics

<layers
=layer_regularization_losses
>non_trainable_variables
1regularization_losses
2trainable_variables
3	variables
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_0/observation/0Placeholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
y
action_0/observation/1Placeholder*'
_output_shapes
:џџџџџџџџџ0*
dtype0*
shape:џџџџџџџџџ0
j
action_0/rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0/step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observation/0action_0/observation/1action_0/rewardaction_0/step_type%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_210438
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
љ
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_210450
к
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_210472

StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_210465
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp+QNetwork/dense_1/kernel/Read/ReadVariableOp)QNetwork/dense_1/bias/Read/ReadVariableOpConst*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_210573
Ё
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_210598кр
Й
,
*__inference_function_with_signature_210468ѕ
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_<lambda>_2632
PartitionedCall*
_input_shapes 
ѕ
d
*__inference_function_with_signature_210457
unknown
identity	ЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_<lambda>_2602
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
Я
H
__inference_<lambda>_260
readvariableop_resource
identity	p
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpY
IdentityIdentityReadVariableOp:value:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:

р
*__inference_function_with_signature_210419
	step_type

reward
discount
observation_0
observation_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_0observation_1unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_polymorphic_action_fn_2104082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ0::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:`\
/
_output_shapes
:џџџџџџџџџ
)
_user_specified_name0/observation/0:XT
'
_output_shapes
:џџџџџџџџџ0
)
_user_specified_name0/observation/1
ЪE
љ
%__inference_polymorphic_action_fn_320
	step_type

reward
discount
observation_0
observation_1A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identityЁ
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2(
&QNetwork/EncodingNetwork/flatten/Constб
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation_0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(QNetwork/EncodingNetwork/flatten/ReshapeЦ
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ 2%
#QNetwork/EncodingNetwork/dense/Castъ
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpё
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%QNetwork/EncodingNetwork/dense/MatMulщ
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp§
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&QNetwork/EncodingNetwork/dense/BiasAddЕ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#QNetwork/EncodingNetwork/dense/ReluР
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOpб
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/MatMulП
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOpХ
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  џ2
Constd
CastCastobservation_1*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ02
Cast
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionЏ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xД
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2Щ
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsЯ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЊ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЮ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1а
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :/2
clip_by_value/Minimum/yВ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ0:::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:^Z
/
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameobservation/0:VR
'
_output_shapes
:џџџџџџџџџ0
'
_user_specified_nameobservation/1

3
!__inference_get_initial_state_254

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ч

__inference__traced_save_210573
file_prefix'
#savev2_variable_read_readvariableop	D
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop6
2savev2_qnetwork_dense_1_kernel_read_readvariableop4
0savev2_qnetwork_dense_1_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e7e4ef0b8b644aef8a109275810a946b/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slicesЬ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop2savev2_qnetwork_dense_1_kernel_read_readvariableop0savev2_qnetwork_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*9
_input_shapes(
&: : : d:d:d0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

: d: 

_output_shapes
:d:$ 

_output_shapes

:d0: 

_output_shapes
:0:

_output_shapes
: 

к
$__inference_signature_wrapper_210438
discount
observation_0
observation_1

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_0observation_1unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *3
f.R,
*__inference_function_with_signature_2104192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ0:џџџџџџџџџ:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:`\
/
_output_shapes
:џџџџџџџџџ
)
_user_specified_name0/observation/0:XT
'
_output_shapes
:џџџџџџџџџ0
)
_user_specified_name0/observation/1:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type
ц'
џ
+__inference_polymorphic_distribution_fn_357
	step_type

reward
discount
observation_0
observation_1A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identityЁ
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2(
&QNetwork/EncodingNetwork/flatten/Constб
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation_0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(QNetwork/EncodingNetwork/flatten/ReshapeЦ
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ 2%
#QNetwork/EncodingNetwork/dense/Castъ
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpё
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%QNetwork/EncodingNetwork/dense/MatMulщ
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp§
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&QNetwork/EncodingNetwork/dense/BiasAddЕ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#QNetwork/EncodingNetwork/dense/ReluР
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOpб
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/MatMulП
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOpХ
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  џ2
Constd
CastCastobservation_1*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ02
Cast
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionЏ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtolk
IdentityIdentityCategorical_1/mode/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ0:::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:^Z
/
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameobservation/0:VR
'
_output_shapes
:џџџџџџџџџ0
'
_user_specified_nameobservation/1

^
$__inference_signature_wrapper_210465
unknown
identity	ЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *3
f.R,
*__inference_function_with_signature_2104572
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
ЩE

(__inference_polymorphic_action_fn_210408
	time_step
time_step_1
time_step_2
time_step_3
time_step_4A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identityЁ
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2(
&QNetwork/EncodingNetwork/flatten/ConstЯ
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(QNetwork/EncodingNetwork/flatten/ReshapeЦ
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ 2%
#QNetwork/EncodingNetwork/dense/Castъ
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpё
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%QNetwork/EncodingNetwork/dense/MatMulщ
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp§
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&QNetwork/EncodingNetwork/dense/BiasAddЕ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#QNetwork/EncodingNetwork/dense/ReluР
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOpб
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/MatMulП
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOpХ
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  џ2
Constb
CastCasttime_step_4*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ02
Cast
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionЏ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xД
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2Щ
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsЯ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЊ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЮ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1а
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :/2
clip_by_value/Minimum/yВ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ0:::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:ZV
/
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:RN
'
_output_shapes
:џџџџџџџџџ0
#
_user_specified_name	time_step
Ќ
<
*__inference_function_with_signature_210445

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_get_initial_state_2104442
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Х
&
$__inference_signature_wrapper_210472
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *3
f.R,
*__inference_function_with_signature_2104682
PartitionedCall*
_input_shapes 
/

__inference_<lambda>_263*
_input_shapes 

6
$__inference_get_initial_state_210444

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ХF
Ў
(__inference_polymorphic_action_fn_210529
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_0
time_step_observation_1A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identityЁ
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2(
&QNetwork/EncodingNetwork/flatten/Constл
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observation_0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(QNetwork/EncodingNetwork/flatten/ReshapeЦ
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ 2%
#QNetwork/EncodingNetwork/dense/Castъ
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpё
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%QNetwork/EncodingNetwork/dense/MatMulщ
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp§
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&QNetwork/EncodingNetwork/dense/BiasAddЕ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#QNetwork/EncodingNetwork/dense/ReluР
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOpб
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/MatMulП
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOpХ
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  џ2
Constn
CastCasttime_step_observation_1*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ02
Cast
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionЏ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xД
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2Щ
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsЯ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЊ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЮ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1а
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :/2
clip_by_value/Minimum/yВ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ0:::::X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step/discount:hd
/
_output_shapes
:џџџџџџџџџ
1
_user_specified_nametime_step/observation/0:`\
'
_output_shapes
:џџџџџџџџџ0
1
_user_specified_nametime_step/observation/1
х
І
"__inference__traced_restore_210598
file_prefix
assignvariableop_variable<
8assignvariableop_1_qnetwork_encodingnetwork_dense_kernel:
6assignvariableop_2_qnetwork_encodingnetwork_dense_bias.
*assignvariableop_3_qnetwork_dense_1_kernel,
(assignvariableop_4_qnetwork_dense_1_bias

identity_6ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slicesЩ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Н
AssignVariableOp_1AssignVariableOp8assignvariableop_1_qnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Л
AssignVariableOp_2AssignVariableOp6assignvariableop_2_qnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Џ
AssignVariableOp_3AssignVariableOp*assignvariableop_3_qnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4­
AssignVariableOp_4AssignVariableOp(assignvariableop_4_qnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЯ

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5С

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ќ
6
$__inference_signature_wrapper_210450

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *3
f.R,
*__inference_function_with_signature_2104452
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"ИL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
action
4

0/discount&
action_0/discount:0џџџџџџџџџ
J
0/observation/07
action_0/observation/0:0џџџџџџџџџ
B
0/observation/1/
action_0/observation/1:0џџџџџџџџџ0
0
0/reward$
action_0/reward:0џџџџџџџџџ
6
0/step_type'
action_0/step_type:0џџџџџџџџџ6
action,
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:№d
Э

train_step
metadata
model_variables
_all_assets

signatures

?action
@distribution
Aget_initial_state
Bget_metadata
Cget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
=
0
1
2
	3"
trackable_tuple_wrapper
'

0"
trackable_list_wrapper
`

Daction
Eget_initial_state
Fget_train_step
Gget_metadata"
signature_map
7:5 d2%QNetwork/EncodingNetwork/dense/kernel
1:/d2#QNetwork/EncodingNetwork/dense/bias
):'d02QNetwork/dense_1/kernel
#:!02QNetwork/dense_1/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
У
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_layerј{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ы
_postprocessing_layers
regularization_losses
trainable_variables
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__" 
_tf_keras_layer{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ъ

kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"Ѕ
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
­
layer_metrics
metrics

layers
layer_regularization_losses
 non_trainable_variables
regularization_losses
trainable_variables
	variables
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
#layer_metrics
$metrics

%layers
&layer_regularization_losses
'non_trainable_variables
regularization_losses
trainable_variables
	variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
­
(layer_metrics
)metrics

*layers
+layer_regularization_losses
,non_trainable_variables
regularization_losses
trainable_variables
	variables
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
т
-regularization_losses
.trainable_variables
/	variables
0	keras_api
*N&call_and_return_all_conditional_losses
O__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ф

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
5layer_metrics
6metrics

7layers
8layer_regularization_losses
9non_trainable_variables
-regularization_losses
.trainable_variables
/	variables
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
:layer_metrics
;metrics

<layers
=layer_regularization_losses
>non_trainable_variables
1regularization_losses
2trainable_variables
3	variables
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
(__inference_polymorphic_action_fn_210529
%__inference_polymorphic_action_fn_320Б
ЊВІ
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsЂ
Ђ 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
+__inference_polymorphic_distribution_fn_357Б
ЊВІ
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsЂ
Ђ 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
!__inference_get_initial_state_254І
В
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
__inference_<lambda>_263
B
__inference_<lambda>_260
mBk
$__inference_signature_wrapper_210438
0/discount0/observation/00/observation/10/reward0/step_type
6B4
$__inference_signature_wrapper_210450
batch_size
(B&
$__inference_signature_wrapper_210465
(B&
$__inference_signature_wrapper_210472
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 7
__inference_<lambda>_260Ђ

Ђ 
Њ " 	0
__inference_<lambda>_263Ђ

Ђ 
Њ "Њ N
!__inference_get_initial_state_254)"Ђ
Ђ


batch_size 
Њ "Ђ г
(__inference_polymorphic_action_fn_210529І	ЩЂХ
НЂЙ
БВ­
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџ
observationqЂn
96
time_step/observation/0џџџџџџџџџ
1.
time_step/observation/1џџџџџџџџџ0
Ђ 
Њ "RВO

PolicyStep&
action
actionџџџџџџџџџ
stateЂ 
infoЂ 
%__inference_polymorphic_action_fn_320ѓ	Ђ
Ђ
ўВњ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџl
observation]ЂZ
/,
observation/0џџџџџџџџџ
'$
observation/1џџџџџџџџџ0
Ђ 
Њ "RВO

PolicyStep&
action
actionџџџџџџџџџ
stateЂ 
infoЂ 
+__inference_polymorphic_distribution_fn_357к	Ђ
Ђ
ўВњ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџl
observation]ЂZ
/,
observation/0џџџџџџџџџ
'$
observation/1џџџџџџџџџ0
Ђ 
Њ "ИВД

PolicyStep
actionџћ№сУлЂз
`
CЂ@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*Њ'
%
loc
Identityџџџџџџџџџ
`Њ]

allow_nan_statsp


atol
 

namejDeterministic


rtol
 

validate_argsp _DistributionTypeSpec
stateЂ 
infoЂ 
$__inference_signature_wrapper_210438и	ЂЂ
Ђ 
Њ
.

0/discount 

0/discountџџџџџџџџџ
D
0/observation/01.
0/observation/0џџџџџџџџџ
<
0/observation/1)&
0/observation/1џџџџџџџџџ0
*
0/reward
0/rewardџџџџџџџџџ
0
0/step_type!
0/step_typeџџџџџџџџџ"+Њ(
&
action
actionџџџџџџџџџ_
$__inference_signature_wrapper_21045070Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "Њ X
$__inference_signature_wrapper_2104650Ђ

Ђ 
Њ "Њ

int64
int64 	<
$__inference_signature_wrapper_210472Ђ

Ђ 
Њ "Њ 