import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../../../core/app_export.dart';
import './models/profile_creation_model.dart';
import './bloc/profile_creation_bloc.dart';
import './bloc/profile_creation_event.dart';
import './bloc/profile_creation_state.dart';
import '../../routes/app_routes.dart';

class AccScreen extends StatelessWidget {
  const AccScreen({super.key});

  static Widget builder(BuildContext context) {
    return BlocProvider(
      create: (_) => ProfileCreationBloc()..add(ProfileCreationInitialEvent()),
      child: const AccScreen(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<ProfileCreationBloc, ProfileCreationState>(
      listener: (context, state) async {
        if (state.isProfileCreated) {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            final profile = state.profileCreationModel;
            final role = (profile.selectedRole ?? '').toLowerCase();

            final data = {
              'username': state.usernameController.text,
              'role': profile.selectedRole,
              'branch': profile.selectedBranch,
              'year': profile.selectedYear,
              'position': profile.selectedPosition,
              'updatedAt': FieldValue.serverTimestamp(),
            };

            await FirebaseFirestore.instance
                .collection('users')
                .doc(user.uid)
                .set(data, SetOptions(merge: true));

            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Profile created successfully!")),
            );

            // Navigate based on role
            if (role == 'student') {
              Navigator.pushReplacementNamed(
                  context, AppRoutes.studentJoinPage);
            } else {
              Navigator.pushReplacementNamed(context, AppRoutes.dashboardPage);
            }
          }
        }
      },
      builder: (context, state) {
        final bloc = context.read<ProfileCreationBloc>();
        final model = state.profileCreationModel;

        return Scaffold(
          backgroundColor: appTheme.whiteCustom,
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        "Create Profile",
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                        ),
                      ),
                      ElevatedButton(
                        onPressed: () {
                          final selectedRole =
                              bloc.state.profileCreationModel.selectedRole;
                          if (selectedRole == null || selectedRole.isEmpty) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                  content: Text("Please select a role.")),
                            );
                            return;
                          }
                          bloc.add(CreateProfileEvent());
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF3D00E0),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 10,
                          ),
                        ),
                        child: const Text(
                          "Create",
                          style: TextStyle(color: Colors.white),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.grey.shade300,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: TextField(
                      controller: state.usernameController,
                      decoration: const InputDecoration(
                        border: InputBorder.none,
                        prefixIcon: Icon(Icons.person),
                        hintText: "User name",
                        contentPadding: EdgeInsets.symmetric(vertical: 14),
                      ),
                      onChanged: (value) {
                        bloc.add(UsernameChangedEvent(value));
                      },
                    ),
                  ),
                  const SizedBox(height: 24),
                  _buildRadioOption(context, bloc, model, "Proctor"),
                  _buildRadioOption(context, bloc, model, "Student"),
                  const SizedBox(height: 16),

                  // Show dropdowns based on role
                  if (model.selectedRole == 'Student') ...[
                    DropdownButtonFormField<String>(
                      value: model.selectedBranch,
                      hint: const Text("Select Branch"),
                      items: [
                        'CS',
                        'ISE',
                        'ESE',
                        'EEE',
                        'CIVIL',
                        'MECHANICAL',
                        'IEM',
                        'ETE',
                        'ML',
                        'CHEMICAL',
                        'BIOTECH',
                        'AIML',
                        'AIDS',
                        'CSDS',
                        'CSBS',
                        'CS IOT'
                      ]
                          .map((branch) => DropdownMenuItem(
                                value: branch,
                                child: Text(branch),
                              ))
                          .toList(),
                      onChanged: (branch) {
                        if (branch != null) {
                          bloc.add(BranchSelectedEvent(branch));
                        }
                      },
                    ),
                    const SizedBox(height: 16),
                    DropdownButtonFormField<String>(
                      value: model.selectedYear,
                      hint: const Text("Select Year"),
                      items: ['1st year', '2nd year', '3rd year', '4th year']
                          .map((year) => DropdownMenuItem(
                                value: year,
                                child: Text(year),
                              ))
                          .toList(),
                      onChanged: (year) {
                        if (year != null) {
                          bloc.add(YearSelectedEvent(year));
                        }
                      },
                    ),
                  ] else if (model.selectedRole == 'Proctor') ...[
                    DropdownButtonFormField<String>(
                      value: model.selectedPosition,
                      hint: const Text("Select Position"),
                      items: [
                        'COE',
                        'DEAN FYB',
                        'DEAN INNOVATION',
                        'PRINCIPAL',
                        'VICE PRINCIPAL',
                        'HOD',
                        'OTHERS'
                      ]
                          .map((pos) => DropdownMenuItem(
                                value: pos,
                                child: Text(pos),
                              ))
                          .toList(),
                      onChanged: (position) {
                        if (position != null) {
                          bloc.add(PositionSelectedEvent(position));
                        }
                      },
                    ),
                  ],
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildRadioOption(BuildContext context, ProfileCreationBloc bloc,
      ProfileCreationModel model, String role) {
    return RadioListTile<String>(
      title: Text(role),
      value: role,
      groupValue: model.selectedRole,
      onChanged: (value) {
        if (value != null) {
          bloc.add(RoleSelectedEvent(value));
        }
      },
    );
  }
}
