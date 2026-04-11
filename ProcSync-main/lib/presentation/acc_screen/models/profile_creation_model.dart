class ProfileCreationModel {
  final String? selectedRole;
  final String? selectedBranch;
  final String? selectedYear;
  final String? selectedPosition;

  const ProfileCreationModel({
    this.selectedRole,
    this.selectedBranch,
    this.selectedYear,
    this.selectedPosition,
  });

  ProfileCreationModel copyWith({
    String? selectedRole,
    String? selectedBranch,
    String? selectedYear,
    String? selectedPosition,
  }) {
    return ProfileCreationModel(
      selectedRole: selectedRole ?? this.selectedRole,
      selectedBranch: selectedBranch ?? this.selectedBranch,
      selectedYear: selectedYear ?? this.selectedYear,
      selectedPosition: selectedPosition ?? this.selectedPosition,
    );
  }
}
