®З
Зў
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
≥
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
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878сХ
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
dtype0*
shape:	2Ц*6
shared_name'%QNetwork/EncodingNetwork/dense/kernel
†
9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	2Ц*
dtype0
Я
#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*4
shared_name%#QNetwork/EncodingNetwork/dense/bias
Ш
7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:Ц*
dtype0
Л
QNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Цd*(
shared_nameQNetwork/dense_1/kernel
Д
+QNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/kernel*
_output_shapes
:	Цd*
dtype0
В
QNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameQNetwork/dense_1/bias
{
)QNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/bias*
_output_shapes
:d*
dtype0

NoOpNoOp
Ќ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*И
valueюBы Bф
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
trainable_variables
	variables
regularization_losses
	keras_api
n
_postprocessing_layers
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2
	3

0
1
2
	3
 
≠
layer_metrics

layers
layer_regularization_losses
trainable_variables
metrics
 non_trainable_variables
	variables
regularization_losses

!0
"1

0
1

0
1
 
≠
#layer_metrics

$layers
%layer_regularization_losses
trainable_variables
&metrics
'non_trainable_variables
	variables
regularization_losses

0
	1

0
	1
 
≠
(layer_metrics

)layers
*layer_regularization_losses
trainable_variables
+metrics
,non_trainable_variables
	variables
regularization_losses
 

0
1
 
 
 
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

kernel
bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
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
 
≠
5layer_metrics

6layers
7layer_regularization_losses
-trainable_variables
8metrics
9non_trainable_variables
.	variables
/regularization_losses

0
1

0
1
 
≠
:layer_metrics

;layers
<layer_regularization_losses
1trainable_variables
=metrics
>non_trainable_variables
2	variables
3regularization_losses
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
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Й
action_0/observation/0Placeholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
y
action_0/observation/1Placeholder*'
_output_shapes
:€€€€€€€€€d*
dtype0*
shape:€€€€€€€€€d
j
action_0/rewardPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
m
action_0/step_typePlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Е
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observation/0action_0/observation/1action_0/rewardaction_0/step_type%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_851701
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
щ
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_851713
Џ
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_851735
Х
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_851728
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
К
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
GPU 2J 8В *(
f#R!
__inference__traced_save_851836
°
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_851861ша
У
6
$__inference_get_initial_state_851707

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ѕ
H
__inference_<lambda>_260
readvariableop_resource
identity	Иp
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
–E
щ
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
identityИ°
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2(
&QNetwork/EncodingNetwork/flatten/Const—
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation_0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(QNetwork/EncodingNetwork/flatten/Reshape∆
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€22%
#QNetwork/EncodingNetwork/dense/Castл
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	2Ц*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpт
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2'
%QNetwork/EncodingNetwork/dense/MatMulк
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpю
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2(
&QNetwork/EncodingNetwork/dense/BiasAddґ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2%
#QNetwork/EncodingNetwork/dense/ReluЅ
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Цd*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp—
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/MatMulњ
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp≈
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А€2
Constd
CastCastobservation_1*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€d2
CastП
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2

SelectV2Х
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimensionѓ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
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
Deterministic/rtolС
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xі
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shapeЗ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeГ
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1Г
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2…
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsѕ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis™
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatќ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisГ
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1–
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :c2
clip_by_value/Minimum/y≤
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yМ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€d:::::N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:^Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameobservation/0:VR
'
_output_shapes
:€€€€€€€€€d
'
_user_specified_nameobservation/1
Р
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
м'
€
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
identityИ°
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2(
&QNetwork/EncodingNetwork/flatten/Const—
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation_0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(QNetwork/EncodingNetwork/flatten/Reshape∆
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€22%
#QNetwork/EncodingNetwork/dense/Castл
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	2Ц*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpт
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2'
%QNetwork/EncodingNetwork/dense/MatMulк
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpю
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2(
&QNetwork/EncodingNetwork/dense/BiasAddґ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2%
#QNetwork/EncodingNetwork/dense/ReluЅ
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Цd*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp—
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/MatMulњ
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp≈
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А€2
Constd
CastCastobservation_1*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€d2
CastП
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2

SelectV2Х
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimensionѓ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
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
:€€€€€€€€€2

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
k:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€d:::::N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:^Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameobservation/0:VR
'
_output_shapes
:€€€€€€€€€d
'
_user_specified_nameobservation/1
н
Н
__inference__traced_save_851836
file_prefix'
#savev2_variable_read_readvariableop	D
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop6
2savev2_qnetwork_dense_1_kernel_read_readvariableop4
0savev2_qnetwork_dense_1_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_371293384660424c973be89e5954cdd9/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameА
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Т
valueИBЕB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slicesћ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop2savev2_qnetwork_dense_1_kernel_read_readvariableop0savev2_qnetwork_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*<
_input_shapes+
): : :	2Ц:Ц:	Цd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	2Ц:!

_output_shapes	
:Ц:%!

_output_shapes
:	Цd: 

_output_shapes
:d:

_output_shapes
: 
ђ
<
*__inference_function_with_signature_851708

batch_sizeР
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
GPU 2J 8В *-
f(R&
$__inference_get_initial_state_8517072
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
/

__inference_<lambda>_263*
_input_shapes 
ЋF
Ѓ
(__inference_polymorphic_action_fn_851792
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_0
time_step_observation_1A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identityИ°
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2(
&QNetwork/EncodingNetwork/flatten/Constџ
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observation_0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(QNetwork/EncodingNetwork/flatten/Reshape∆
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€22%
#QNetwork/EncodingNetwork/dense/Castл
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	2Ц*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpт
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2'
%QNetwork/EncodingNetwork/dense/MatMulк
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpю
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2(
&QNetwork/EncodingNetwork/dense/BiasAddґ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2%
#QNetwork/EncodingNetwork/dense/ReluЅ
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Цd*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp—
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/MatMulњ
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp≈
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А€2
Constn
CastCasttime_step_observation_1*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€d2
CastП
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2

SelectV2Х
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimensionѓ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
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
Deterministic/rtolС
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xі
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shapeЗ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeГ
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1Г
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2…
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsѕ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis™
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatќ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisГ
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1–
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :c2
clip_by_value/Minimum/y≤
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yМ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€d:::::X T
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nametime_step/discount:hd
/
_output_shapes
:€€€€€€€€€
1
_user_specified_nametime_step/observation/0:`\
'
_output_shapes
:€€€€€€€€€d
1
_user_specified_nametime_step/observation/1
Щ
а
*__inference_function_with_signature_851682
	step_type

reward
discount
observation_0
observation_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall•
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_0observation_1unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *1
f,R*
(__inference_polymorphic_action_fn_8516712
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€d::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:`\
/
_output_shapes
:€€€€€€€€€
)
_user_specified_name0/observation/0:XT
'
_output_shapes
:€€€€€€€€€d
)
_user_specified_name0/observation/1
ђ
6
$__inference_signature_wrapper_851713

batch_sizeЦ
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
GPU 2J 8В *3
f.R,
*__inference_function_with_signature_8517082
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
е
¶
"__inference__traced_restore_851861
file_prefix
assignvariableop_variable<
8assignvariableop_1_qnetwork_encodingnetwork_dense_kernel:
6assignvariableop_2_qnetwork_encodingnetwork_dense_bias.
*assignvariableop_3_qnetwork_dense_1_kernel,
(assignvariableop_4_qnetwork_dense_1_bias

identity_6ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4Ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Т
valueИBЕB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices…
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

IdentityШ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1љ
AssignVariableOp_1AssignVariableOp8assignvariableop_1_qnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ї
AssignVariableOp_2AssignVariableOp6assignvariableop_2_qnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ѓ
AssignVariableOp_3AssignVariableOp*assignvariableop_3_qnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4≠
AssignVariableOp_4AssignVariableOp(assignvariableop_4_qnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpѕ

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5Ѕ

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
Х
Џ
$__inference_signature_wrapper_851701
discount
observation_0
observation_1

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_0observation_1unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *3
f.R,
*__inference_function_with_signature_8516822
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€d:€€€€€€€€€:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:`\
/
_output_shapes
:€€€€€€€€€
)
_user_specified_name0/observation/0:XT
'
_output_shapes
:€€€€€€€€€d
)
_user_specified_name0/observation/1:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:PL
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type
х
d
*__inference_function_with_signature_851720
unknown
identity	ИҐStatefulPartitionedCall°
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
GPU 2J 8В *!
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
ѕE
А
(__inference_polymorphic_action_fn_851671
	time_step
time_step_1
time_step_2
time_step_3
time_step_4A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identityИ°
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€2   2(
&QNetwork/EncodingNetwork/flatten/Constѕ
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(QNetwork/EncodingNetwork/flatten/Reshape∆
#QNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€22%
#QNetwork/EncodingNetwork/dense/Castл
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	2Ц*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpт
%QNetwork/EncodingNetwork/dense/MatMulMatMul'QNetwork/EncodingNetwork/dense/Cast:y:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2'
%QNetwork/EncodingNetwork/dense/MatMulк
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpю
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2(
&QNetwork/EncodingNetwork/dense/BiasAddґ
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ц2%
#QNetwork/EncodingNetwork/dense/ReluЅ
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Цd*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOp—
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/MatMulњ
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp≈
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
QNetwork/dense_1/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А€2
Constb
CastCasttime_step_4*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€d2
CastП
SelectV2SelectV2Cast:y:0!QNetwork/dense_1/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2

SelectV2Х
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimensionѓ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
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
Deterministic/rtolС
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xі
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shapeЗ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeГ
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1Г
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2…
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsѕ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis™
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatќ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisГ
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1–
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :c2
clip_by_value/Minimum/y≤
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yМ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€d:::::N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:ZV
/
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:RN
'
_output_shapes
:€€€€€€€€€d
#
_user_specified_name	time_step
Б
^
$__inference_signature_wrapper_851728
unknown
identity	ИҐStatefulPartitionedCall≥
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
GPU 2J 8В *3
f.R,
*__inference_function_with_signature_8517202
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
є
,
*__inference_function_with_signature_851731х
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
GPU 2J 8В *!
fR
__inference_<lambda>_2632
PartitionedCall*
_input_shapes 
≈
&
$__inference_signature_wrapper_851735З
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
GPU 2J 8В *3
f.R,
*__inference_function_with_signature_8517312
PartitionedCall*
_input_shapes "ЄL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*П
actionД
4

0/discount&
action_0/discount:0€€€€€€€€€
J
0/observation/07
action_0/observation/0:0€€€€€€€€€
B
0/observation/1/
action_0/observation/1:0€€€€€€€€€d
0
0/reward$
action_0/reward:0€€€€€€€€€
6
0/step_type'
action_0/step_type:0€€€€€€€€€6
action,
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:фd
Ќ
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
8:6	2Ц2%QNetwork/EncodingNetwork/dense/kernel
2:0Ц2#QNetwork/EncodingNetwork/dense/bias
*:(	Цd2QNetwork/dense_1/kernel
#:!d2QNetwork/dense_1/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
√
_encoder
_q_value_layer
trainable_variables
	variables
regularization_losses
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"Т
_tf_keras_layerш{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ћ
_postprocessing_layers
trainable_variables
	variables
regularization_losses
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"†
_tf_keras_layerЖ{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ћ

kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"¶
_tf_keras_layerМ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 150]}}
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
 "
trackable_list_wrapper
≠
layer_metrics

layers
layer_regularization_losses
trainable_variables
metrics
 non_trainable_variables
	variables
regularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
#layer_metrics

$layers
%layer_regularization_losses
trainable_variables
&metrics
'non_trainable_variables
	variables
regularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
(layer_metrics

)layers
*layer_regularization_losses
trainable_variables
+metrics
,non_trainable_variables
	variables
regularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
в
-trainable_variables
.	variables
/regularization_losses
0	keras_api
N__call__
*O&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ƒ

kernel
bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 50]}}
 "
trackable_dict_wrapper
.
!0
"1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
5layer_metrics

6layers
7layer_regularization_losses
-trainable_variables
8metrics
9non_trainable_variables
.	variables
/regularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
:layer_metrics

;layers
<layer_regularization_losses
1trainable_variables
=metrics
>non_trainable_variables
2	variables
3regularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
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
И2Е
(__inference_polymorphic_action_fn_851792
%__inference_polymorphic_action_fn_320±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
д2б
+__inference_polymorphic_distribution_fn_357±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
!__inference_get_initial_state_254¶
Э≤Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
B
__inference_<lambda>_263
B
__inference_<lambda>_260
mBk
$__inference_signature_wrapper_851701
0/discount0/observation/00/observation/10/reward0/step_type
6B4
$__inference_signature_wrapper_851713
batch_size
(B&
$__inference_signature_wrapper_851728
(B&
$__inference_signature_wrapper_851735
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 7
__inference_<lambda>_260Ґ

Ґ 
™ "К 	0
__inference_<lambda>_263Ґ

Ґ 
™ "™ N
!__inference_get_initial_state_254)"Ґ
Ґ
К

batch_size 
™ "Ґ Э
%__inference_polymorphic_action_fn_320у	ЦҐТ
КҐЖ
ю≤ъ
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€l
observation]ҐZ
/К,
observation/0€€€€€€€€€
'К$
observation/1€€€€€€€€€d
Ґ 
™ "R≤O

PolicyStep&
actionК
action€€€€€€€€€
stateҐ 
infoҐ ”
(__inference_polymorphic_action_fn_851792¶	…Ґ≈
љҐє
±≤≠
TimeStep6
	step_type)К&
time_step/step_type€€€€€€€€€0
reward&К#
time_step/reward€€€€€€€€€4
discount(К%
time_step/discount€€€€€€€€€А
observationqҐn
9К6
time_step/observation/0€€€€€€€€€
1К.
time_step/observation/1€€€€€€€€€d
Ґ 
™ "R≤O

PolicyStep&
actionК
action€€€€€€€€€
stateҐ 
infoҐ К
+__inference_polymorphic_distribution_fn_357Џ	ЦҐТ
КҐЖ
ю≤ъ
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€l
observation]ҐZ
/К,
observation/0€€€€€€€€€
'К$
observation/1€€€€€€€€€d
Ґ 
™ "Є≤і

PolicyStepК
action€Тырб√ГџҐ„
`
CҐ@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*™'
%
locК
Identity€€€€€€€€€
`™]
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
stateҐ 
infoҐ Б
$__inference_signature_wrapper_851701Ў	ҐҐЮ
Ґ 
Ц™Т
.

0/discount К

0/discount€€€€€€€€€
D
0/observation/01К.
0/observation/0€€€€€€€€€
<
0/observation/1)К&
0/observation/1€€€€€€€€€d
*
0/rewardК
0/reward€€€€€€€€€
0
0/step_type!К
0/step_type€€€€€€€€€"+™(
&
actionК
action€€€€€€€€€_
$__inference_signature_wrapper_85171370Ґ-
Ґ 
&™#
!

batch_sizeК

batch_size "™ X
$__inference_signature_wrapper_8517280Ґ

Ґ 
™ "™

int64К
int64 	<
$__inference_signature_wrapper_851735Ґ

Ґ 
™ "™ 