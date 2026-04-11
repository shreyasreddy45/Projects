import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter/material.dart';
import 'profile_creation_event.dart';
import 'profile_creation_state.dart';
import '../models/profile_creation_model.dart';

class ProfileCreationBloc
    extends Bloc<ProfileCreationEvent, ProfileCreationState> {
  ProfileCreationBloc()
      : super(ProfileCreationState(
          usernameController: TextEditingController(),
          profileCreationModel: const ProfileCreationModel(),
        )) {
    on<ProfileCreationInitialEvent>(_onInit);
    on<UsernameChangedEvent>(_onUsernameChanged);
    on<RoleSelectedEvent>(_onRoleSelected);
    on<BranchSelectedEvent>(_onBranchSelected);
    on<YearSelectedEvent>(_onYearSelected);
    on<PositionSelectedEvent>(_onPositionSelected);
    on<CreateProfileEvent>(_onCreateProfile);
  }

  void _onInit(
    ProfileCreationInitialEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    emit(ProfileCreationState(
      usernameController: TextEditingController(),
      profileCreationModel: const ProfileCreationModel(),
    ));
  }

  void _onUsernameChanged(
    UsernameChangedEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    state.usernameController.text = event.username;
    emit(state.copyWith());
  }

  void _onRoleSelected(
    RoleSelectedEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    emit(state.copyWith(
      profileCreationModel: state.profileCreationModel.copyWith(
        selectedRole: event.role,
        selectedBranch: null,
        selectedYear: null,
        selectedPosition: null,
      ),
    ));
  }

  void _onBranchSelected(
    BranchSelectedEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    emit(state.copyWith(
      profileCreationModel:
          state.profileCreationModel.copyWith(selectedBranch: event.branch),
    ));
  }

  void _onYearSelected(
    YearSelectedEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    emit(state.copyWith(
      profileCreationModel:
          state.profileCreationModel.copyWith(selectedYear: event.year),
    ));
  }

  void _onPositionSelected(
    PositionSelectedEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    emit(state.copyWith(
      profileCreationModel:
          state.profileCreationModel.copyWith(selectedPosition: event.position),
    ));
  }

  void _onCreateProfile(
    CreateProfileEvent event,
    Emitter<ProfileCreationState> emit,
  ) {
    emit(state.copyWith(isProfileCreated: true));
    // You can add navigation or API call here
  }
}
