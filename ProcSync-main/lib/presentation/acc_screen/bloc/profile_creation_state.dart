import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import '../models/profile_creation_model.dart';

class ProfileCreationState extends Equatable {
  final TextEditingController usernameController;
  final ProfileCreationModel profileCreationModel;
  final bool isProfileCreated;

  const ProfileCreationState({
    required this.usernameController,
    required this.profileCreationModel,
    this.isProfileCreated = false,
  });

  ProfileCreationState copyWith({
    TextEditingController? usernameController,
    ProfileCreationModel? profileCreationModel,
    bool? isProfileCreated,
  }) {
    return ProfileCreationState(
      usernameController: usernameController ?? this.usernameController,
      profileCreationModel: profileCreationModel ?? this.profileCreationModel,
      isProfileCreated: isProfileCreated ?? this.isProfileCreated,
    );
  }

  @override
  List<Object?> get props =>
      [usernameController, profileCreationModel, isProfileCreated];
}
